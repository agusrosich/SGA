"""Censoring-aware causal survival analysis for the SEER-derived SGA cohort.

Primary estimand: ATE in 10-year RMST (chemotherapy recorded vs no chemotherapy)
among T3/T4 patients with recorded radiotherapy. SEER fields in this extract do
not establish surgery, postoperative intent, concurrent timing, drug, dose, M
stage, grade, margins, ENE, PNI, performance status, diagnosis year, or gland
site. The analysis therefore is RTOG 1008-like, not a strict trial emulation.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TAU = 120
HORIZONS = (60, 120)
CLINICALLY_RELEVANT_RMST_MONTHS = 6.0
CLINICALLY_RELEVANT_SURVIVAL_DIFFERENCE = 0.05
CATEGORICAL = ["Sex", "T_Unified", "N_Unified", "Histology_Unified"]
NUMERIC = ["Age_Median"]
COVARIATES = NUMERIC + CATEGORICAL
COMPARABLE_HISTOLOGIES = {
    "Mucoepidermoid carcinoma", "Adenocarcinoma", "Acinar cell carcinoma",
    "Adenoid cystic carcinoma", "Carcinoma NOS",
    "Carcinoma ex pleomorphic adenoma",
}


def encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", encoder())]), CATEGORICAL),
    ])


def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    d = raw.copy()
    d["Treatment"] = d["Chemotherapy_Binary"].astype(int)
    d["OS_Event"] = d["COD to site recode"].ne("Alive").astype(int)
    d["CSS_Event"] = d["COD to site recode"].eq("Salivary Gland").astype(int)
    d["Time"] = pd.to_numeric(d["Survival months"], errors="coerce").clip(lower=0)
    d["Time_120"] = d["Time"].clip(upper=TAU)
    d["OS_Event_120"] = ((d["OS_Event"] == 1) & (d["Time"] <= TAU)).astype(int)
    d["CSS_Event_120"] = ((d["CSS_Event"] == 1) & (d["Time"] <= TAU)).astype(int)
    d["T34"] = d["T_Unified"].astype(str).str.startswith(("T3", "T4"))
    d["Known_N"] = ~d["N_Unified"].isin(["NX", "N88"])
    d["Comparable_Histology"] = d["Histology_Unified"].isin(COMPARABLE_HISTOLOGIES)
    d["Squamous"] = d["Histology_Unified"].astype(str).str.contains("squamous", case=False)
    flow = []
    def add(step: str, mask: pd.Series) -> None:
        flow.append({"Step": step, "Remaining_N": int(mask.sum()), "Excluded_at_step_N": 0})
    mask = pd.Series(True, index=d.index); add("Source standardized records", mask)
    previous = int(mask.sum()); mask &= d["T34"]; add("T3/T4 disease", mask); flow[-1]["Excluded_at_step_N"] = previous-int(mask.sum())
    previous = int(mask.sum()); mask &= d["Known_N"]; add("Known N category", mask); flow[-1]["Excluded_at_step_N"] = previous-int(mask.sum())
    previous = int(mask.sum()); mask &= d["Radiation_Binary"].eq(1); add("Radiotherapy recorded", mask); flow[-1]["Excluded_at_step_N"] = previous-int(mask.sum())
    previous = int(mask.sum()); mask &= d["Time"].notna(); add("Nonmissing survival time", mask); flow[-1]["Excluded_at_step_N"] = previous-int(mask.sum())
    return d.loc[mask].copy().reset_index(drop=True), pd.DataFrame(flow)


def design_matrix_fit(d: pd.DataFrame) -> tuple[ColumnTransformer, pd.DataFrame]:
    prep = preprocessor()
    x = prep.fit_transform(d[COVARIATES])
    names = [n.replace("num__", "").replace("cat__", "") for n in prep.get_feature_names_out()]
    return prep, pd.DataFrame(x, columns=names, index=d.index)


def propensity_and_weights(d: pd.DataFrame) -> tuple[pd.DataFrame, Pipeline]:
    model = Pipeline([("prep", preprocessor()), ("logit", LogisticRegression(max_iter=4000, C=1.0))])
    model.fit(d[COVARIATES], d["Treatment"])
    out = d.copy()
    out["Propensity"] = np.clip(model.predict_proba(d[COVARIATES])[:, 1], .01, .99)
    p = out["Treatment"].mean()
    out["IPTW_raw"] = np.where(out["Treatment"].eq(1), p/out["Propensity"], (1-p)/(1-out["Propensity"]))
    lo, hi = out["IPTW_raw"].quantile([.01, .99])
    out["IPTW"] = out["IPTW_raw"].clip(lo, hi)
    return out, model


def smd_table(d: pd.DataFrame) -> pd.DataFrame:
    _, x = design_matrix_fit(d)
    rows = []
    for col in x.columns:
        z = x[col].to_numpy(float); a = d["Treatment"].to_numpy(int)
        vals = {}
        for label, w in [("Before", np.ones(len(d))), ("After_IPTW", d["IPTW"].to_numpy(float))]:
            m1=np.average(z[a==1], weights=w[a==1]); m0=np.average(z[a==0], weights=w[a==0])
            v1=np.average((z[a==1]-m1)**2, weights=w[a==1]); v0=np.average((z[a==0]-m0)**2, weights=w[a==0])
            vals[label]=(m1-m0)/np.sqrt((v1+v0)/2) if v1+v0 else 0.0
        rows.append({"Covariate": col, "SMD_Before": vals["Before"], "SMD_After_IPTW": vals["After_IPTW"]})
    return pd.DataFrame(rows).sort_values("SMD_Before", key=lambda s:s.abs(), ascending=False)


def fit_weighted_cox(d: pd.DataFrame, event: str = "OS_Event_120") -> tuple[CoxPHFitter, pd.DataFrame, ColumnTransformer]:
    prep, x = design_matrix_fit(d)
    fit = x.copy(); fit.insert(0, "Treatment", d["Treatment"].to_numpy())
    fit["Time_120"] = d["Time_120"].to_numpy(); fit[event] = d[event].to_numpy(); fit["IPTW"] = d["IPTW"].to_numpy()
    cph = CoxPHFitter(penalizer=.02)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(fit, "Time_120", event, weights_col="IPTW", robust=True)
    return cph, fit, prep


def marginal_curves(cph: CoxPHFitter, fit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = np.arange(0, TAU+1, 1, dtype=float)
    covs = fit.drop(columns=["Time_120", "OS_Event_120", "CSS_Event_120", "IPTW"], errors="ignore")
    curves = [] ; individual = pd.DataFrame(index=fit.index)
    for arm in (0, 1):
        cf = covs.copy(); cf["Treatment"] = arm
        s = cph.predict_survival_function(cf, times=grid).to_numpy()
        mean_s = s.mean(axis=1)
        curves.append(pd.DataFrame({"Month":grid, "Survival":mean_s, "Treatment":arm}))
        individual[f"RMST_{arm}"] = np.trapezoid(s, grid, axis=0)
        for h in HORIZONS: individual[f"S{h}_{arm}"] = s[int(h), :]
    individual["RMST_Difference"] = individual["RMST_1"]-individual["RMST_0"]
    return pd.concat(curves, ignore_index=True), individual


def estimate(d: pd.DataFrame, event: str = "OS_Event_120") -> dict:
    weighted, _ = propensity_and_weights(d)
    cph, fit, _ = fit_weighted_cox(weighted, event)
    curves, indiv = marginal_curves(cph, fit)
    c0=curves[curves.Treatment.eq(0)].set_index("Month").Survival
    c1=curves[curves.Treatment.eq(1)].set_index("Month").Survival
    return {"data":weighted, "cox":cph, "fit":fit, "curves":curves, "individual":indiv,
            "rmst0":float(indiv.RMST_0.mean()), "rmst1":float(indiv.RMST_1.mean()),
            "delta_rmst":float(indiv.RMST_Difference.mean()),
            "s60_0":float(c0.loc[60]), "s60_1":float(c1.loc[60]),
            "s120_0":float(c0.loc[120]), "s120_1":float(c1.loc[120])}


def bootstrap(d: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng=np.random.default_rng(seed); rows=[]
    for i in range(n):
        b=d.iloc[rng.integers(0,len(d),len(d))].reset_index(drop=True)
        try:
            e=estimate(b)
            rows.append({"Iteration":i+1,"Delta_RMST_Months":e["delta_rmst"],
                         "Survival_Difference_5y":e["s60_1"]-e["s60_0"],
                         "Survival_Difference_10y":e["s120_1"]-e["s120_0"]})
        except Exception:
            rows.append({"Iteration":i+1,"Delta_RMST_Months":np.nan,
                         "Survival_Difference_5y":np.nan,"Survival_Difference_10y":np.nan})
    return pd.DataFrame(rows)


def reverse_km_followup(d: pd.DataFrame) -> float:
    km=KaplanMeierFitter().fit(d["Time"], event_observed=1-d["OS_Event"])
    return float(km.median_survival_time_)


def validation(d: pd.DataFrame, folds: int, seed: int) -> pd.DataFrame:
    rows=[]; kf=KFold(folds, shuffle=True, random_state=seed)
    for fold,(tr,te) in enumerate(kf.split(d),1):
        train,_=propensity_and_weights(d.iloc[tr].reset_index(drop=True)); test=d.iloc[te].reset_index(drop=True)
        cph,_,prep=fit_weighted_cox(train)
        x=pd.DataFrame(prep.transform(test[COVARIATES]), columns=[n.replace("num__","").replace("cat__","") for n in prep.get_feature_names_out()])
        x.insert(0,"Treatment",test.Treatment.to_numpy())
        risk=cph.predict_partial_hazard(x).to_numpy().ravel()
        ci=concordance_index(test.Time_120, -risk, test.OS_Event_120)
        row={"Fold":fold,"C_index":ci}
        censor_km=KaplanMeierFitter().fit(train.Time_120, event_observed=1-train.OS_Event_120)
        for h in HORIZONS:
            pred=1-cph.predict_survival_function(x,times=[h]).to_numpy().ravel()
            y=((test.OS_Event_120.eq(1))&(test.Time_120<=h)).astype(int)
            weights=np.zeros(len(test),dtype=float)
            event_before=(test.OS_Event_120.eq(1)&test.Time_120.le(h)).to_numpy()
            at_risk=test.Time_120.ge(h).to_numpy()
            if event_before.any():
                g=np.clip(censor_km.predict(np.maximum(test.loc[event_before,"Time_120"].to_numpy()-.001,0)).to_numpy(),.01,1)
                weights[event_before]=1/g
            weights[at_risk]=1/max(float(censor_km.predict(max(h-.001,0))),.01)
            row[f"Brier_{h}m"] = float(np.mean(weights*(y.to_numpy()-pred)**2))
            km=KaplanMeierFitter().fit(test.Time_120, test.OS_Event_120)
            observed=1-float(km.predict(h)); row[f"Calibration_Error_{h}m"]=float(pred.mean()-observed)
        rows.append(row)
    out=pd.DataFrame(rows); out.loc[len(out)]={"Fold":"Mean",**{c:out[c].mean() for c in out.columns if c!="Fold"}}
    return out


def baseline_table(d: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for arm,label in [(0,"RT only"),(1,"RT + chemotherapy")]:
        z=d[d.Treatment.eq(arm)]; rows.append({"Treatment_Group":label,"N":len(z),"Age_Median":z.Age_Median.median(),
            "Male_Percent":z.Sex.eq("Male").mean()*100,"T4_Percent":z.T_Unified.str.startswith("T4").mean()*100,
            "Node_Positive_Percent":z.N_Unified.ne("N0").mean()*100,"Deaths_Percent":z.OS_Event.mean()*100})
    return pd.DataFrame(rows)


def available_treatment_timing(d: pd.DataFrame) -> pd.DataFrame:
    delay=pd.to_numeric(d["Time from diagnosis to treatment in days recode"],errors="coerce")
    rows=[]
    for arm,label in [(0,"RT only"),(1,"RT + chemotherapy")]:
        z=delay[d.Treatment.eq(arm)].dropna()
        rows.append({"Group":label,"N_with_calculable_delay":len(z),"Missing_or_uncalculable_N":int(d.Treatment.eq(arm).sum()-len(z)),
            "Median_days_diagnosis_to_first_treatment":z.median(),"Q1":z.quantile(.25),"Q3":z.quantile(.75)})
    return pd.DataFrame(rows)


def sensitivity(d: pd.DataFrame) -> pd.DataFrame:
    specs={"Primary":pd.Series(True,index=d.index),"Exclude squamous carcinoma":~d.Squamous,
           "RTOG 1008-comparable histologies":d.Comparable_Histology}
    rows=[]
    for name,mask in specs.items():
        z=d[mask].reset_index(drop=True)
        try:
            e=estimate(z); rows.append({"Analysis":name,"N":len(z),"Delta_RMST_Months":e["delta_rmst"],
                "Survival_Difference_5y":e["s60_1"]-e["s60_0"],"Survival_Difference_10y":e["s120_1"]-e["s120_0"]})
        except Exception: rows.append({"Analysis":name,"N":len(z)})
    return pd.DataFrame(rows)


def histology_sensitivity(d: pd.DataFrame, minimum_n: int = 80) -> pd.DataFrame:
    """Prespecified descriptive sensitivity estimates; no subgroup p-values."""
    rows=[]
    for hist,z in d.groupby("Histology_Unified"):
        if len(z)<minimum_n or z.Treatment.nunique()<2 or z.Treatment.value_counts().min()<15:
            continue
        try:
            e=estimate(z.reset_index(drop=True)); rows.append({"Histology":hist,"N":len(z),
                "Treated_N":int(z.Treatment.sum()),"Delta_RMST_Months":e["delta_rmst"],
                "Survival_Difference_5y":e["s60_1"]-e["s60_0"]})
        except Exception:
            continue
    return pd.DataFrame(rows)


def simulated_trials(d: pd.DataFrame, indiv: pd.DataFrame, repetitions: int, seed: int, n: int=252) -> pd.DataFrame:
    rng=np.random.default_rng(seed); rows=[]; m=len(indiv)
    balance=pd.DataFrame({"Age":d.Age_Median,"Male":d.Sex.eq("Male").astype(int),
        "T4":d.T_Unified.str.startswith("T4").astype(int),"Node_positive":d.N_Unified.ne("N0").astype(int)})
    for i in range(repetitions):
        idx=rng.choice(m,n,replace=m<n); arm=np.array([0]*(n//2)+[1]*(n-n//2)); rng.shuffle(arm)
        sample=indiv.iloc[idx].reset_index(drop=True)
        outcome=np.where(arm==1,sample.RMST_1,sample.RMST_0)
        b=balance.iloc[idx].reset_index(drop=True); smds=[]
        for col in b:
            x=b[col].to_numpy(float); m1=x[arm==1].mean(); m0=x[arm==0].mean(); den=np.sqrt((x[arm==1].var()+x[arm==0].var())/2)
            smds.append((m1-m0)/den if den else 0)
        rows.append({"Simulation":i+1,"N":n,"Delta_RMST_Months":outcome[arm==1].mean()-outcome[arm==0].mean(),
            "SMD_Age":smds[0],"SMD_Male":smds[1],"SMD_T4":smds[2],"SMD_Node_Positive":smds[3],"Maximum_Absolute_SMD":max(map(abs,smds))})
    return pd.DataFrame(rows)


def evalue_table(cph: CoxPHFitter) -> pd.DataFrame:
    """Approximate E-value sensitivity analysis for the treatment hazard ratio."""
    hr=float(np.exp(cph.params_["Treatment"]))
    ci=cph.confidence_intervals_.loc["Treatment"].to_numpy(float)
    lo,hi=np.exp(ci[0]),np.exp(ci[1])
    def ev(r: float) -> float:
        r=max(r,1/r)
        return float(r+np.sqrt(r*(r-1)))
    closest=lo if hr>1 else hi
    bound=1.0 if lo<=1<=hi else ev(float(closest))
    return pd.DataFrame([{"Treatment_Hazard_Ratio":hr,"CI95_Lower":lo,"CI95_Upper":hi,"Robust_Wald_P_Value":float(cph.summary.loc["Treatment","p"]),
        "E_Value_Point_Estimate":ev(hr),"E_Value_CI_Bound":bound,
        "Interpretation":"Minimum risk-ratio association an unmeasured confounder would need with treatment and outcome, conditional on measured covariates, to explain the estimate."}])


def plots(out: Path, e: dict, boot: pd.DataFrame, smd: pd.DataFrame) -> None:
    out.mkdir(parents=True,exist_ok=True); plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    fig,ax=plt.subplots(figsize=(7,4.5))
    for arm,label,color in [(0,"RT only","#475569"),(1,"RT + chemotherapy","#2563eb")]:
        z=e["curves"].query("Treatment==@arm"); ax.plot(z.Month,z.Survival,label=label,color=color,lw=2.2)
    ax.set(xlabel="Months", ylabel="Adjusted marginal survival probability", ylim=(0,1.02), xlim=(0,TAU)); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(out/"adjusted_marginal_survival.png",dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5)); z=boot.Delta_RMST_Months.dropna(); ax.hist(z,bins=30,color="#0f766e",alpha=.85); ax.axvline(0,color="black",ls="--"); ax.set(xlabel="ATE: 10-year RMST difference (months)",ylabel="Bootstrap iterations"); ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(out/"bootstrap_rmst_difference.png",dpi=300); plt.close(fig)
    top=smd.assign(abs_before=smd.SMD_Before.abs()).nlargest(20,"abs_before").sort_values("abs_before")
    fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(top.SMD_Before,top.Covariate,label="Before",color="#dc2626"); ax.scatter(top.SMD_After_IPTW,top.Covariate,label="After IPTW",color="#2563eb"); ax.axvline(.1,color="grey",ls="--"); ax.axvline(-.1,color="grey",ls="--"); ax.set_xlabel("Standardized mean difference"); ax.legend(); fig.tight_layout(); fig.savefig(out/"covariate_balance_love_plot.png",dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5));
    for arm,label,color in [(0,"RT only","#475569"),(1,"RT + chemotherapy","#2563eb")]:
        ax.hist(e["data"].loc[e["data"].Treatment.eq(arm),"Propensity"],bins=25,density=True,alpha=.45,label=label,color=color)
    ax.set(xlabel="Propensity score",ylabel="Density"); ax.legend(); fig.tight_layout(); fig.savefig(out/"propensity_overlap.png",dpi=300); plt.close(fig)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="data/raw/ExportadaSEER_Estandarizada.csv"); ap.add_argument("--bootstrap",type=int,default=1000); ap.add_argument("--simulations",type=int,default=1000); ap.add_argument("--seed",type=int,default=SEED); ap.add_argument("--reuse-bootstrap",action="store_true",help="Reuse an existing complete bootstrap table while regenerating other outputs."); args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]; out=root/"outputs"/"causal_survival"; tables=out/"tables"; figs=out/"figures"; tables.mkdir(parents=True,exist_ok=True)
    d,flow=load_data(root/args.input); e=estimate(d)
    bootstrap_path=tables/"bootstrap.csv"
    if args.reuse_bootstrap and bootstrap_path.exists():
        boot=pd.read_csv(bootstrap_path)
    else:
        boot=bootstrap(d,args.bootstrap,args.seed)
    smd=smd_table(e["data"]); valid=validation(d,5,args.seed); sens=sensitivity(d); sims=simulated_trials(d,e["individual"],args.simulations,args.seed)
    ci=np.nanpercentile(boot.Delta_RMST_Months,[2.5,97.5])
    delta5=e["s60_1"]-e["s60_0"]; delta10=e["s120_1"]-e["s120_0"]
    def bootstrap_wald_p(estimate: float, column: str) -> float:
        se=boot[column].std(ddof=1)
        return float(2*stats.norm.sf(abs(estimate/se)))
    result=pd.DataFrame([{"Estimand":"ATE: difference in 10-year RMST","Population":"T3/T4, known N, radiotherapy recorded","N":len(d),"Treated_N":int(d.Treatment.sum()),"Control_N":int((1-d.Treatment).sum()),"RMST_No_Chemotherapy_Months":e["rmst0"],"RMST_Chemotherapy_Months":e["rmst1"],"Delta_RMST_Months":e["delta_rmst"],"CI95_Lower":ci[0],"CI95_Upper":ci[1],"P_Value_RMST":bootstrap_wald_p(e["delta_rmst"],"Delta_RMST_Months"),"Survival_Difference_5y":delta5,"P_Value_Survival_5y":bootstrap_wald_p(delta5,"Survival_Difference_5y"),"Survival_Difference_10y":delta10,"P_Value_Survival_10y":bootstrap_wald_p(delta10,"Survival_Difference_10y"),"Reverse_KM_Median_Followup_Months":reverse_km_followup(d),"Bootstrap_Successful":int(boot.Delta_RMST_Months.notna().sum())}])
    missing=pd.DataFrame({"Variable":d.columns,"Missing_N":[d[c].isna().sum() for c in d.columns],"Missing_Percent":[d[c].isna().mean()*100 for c in d.columns]})
    ps1=e["data"].query("Treatment==1").Propensity; ps0=e["data"].query("Treatment==0").Propensity
    common_lo=max(ps1.quantile(.01),ps0.quantile(.01)); common_hi=min(ps1.quantile(.99),ps0.quantile(.99))
    positivity=pd.DataFrame([{"Group":"RT + chemotherapy","N":len(ps1),"PS_Min":ps1.min(),"PS_P1":ps1.quantile(.01),"PS_P99":ps1.quantile(.99),"PS_Max":ps1.max(),"Percent_in_1_99_Common_Support":ps1.between(common_lo,common_hi).mean()*100},{"Group":"RT only","N":len(ps0),"PS_Min":ps0.min(),"PS_P1":ps0.quantile(.01),"PS_P99":ps0.quantile(.99),"PS_Max":ps0.max(),"Percent_in_1_99_Common_Support":ps0.between(common_lo,common_hi).mean()*100}])
    css=estimate(d,event="CSS_Event_120"); css_result=pd.DataFrame([{"Endpoint":"Cancer-specific mortality (separate secondary cause-specific model)","Delta_RMST_Months":css["delta_rmst"],"Survival_Difference_5y":css["s60_1"]-css["s60_0"],"Survival_Difference_10y":css["s120_1"]-css["s120_0"]}])
    outputs={"primary_result":result,"cohort_flow":flow,"baseline_observed_groups":baseline_table(d),"available_treatment_timing":available_treatment_timing(d),"balance":smd,"positivity":positivity,"bootstrap":boot,"validation":valid,"sensitivity":sens,"histology_sensitivity":histology_sensitivity(d),"unmeasured_confounding_evalue":evalue_table(e["cox"]),"simulated_trials_252":sims,"missingness":missing,"cancer_specific_secondary":css_result,"adjusted_survival_curves":e["curves"]}
    for name,frame in outputs.items(): frame.to_csv(tables/f"{name}.csv",index=False)
    plots(figs,e,boot,smd)
    metadata={"primary_endpoint":"Overall survival with living patients censored at last follow-up","estimand":"ATE difference in RMST through 120 months","prespecified_clinical_relevance":{"absolute_RMST_months":CLINICALLY_RELEVANT_RMST_MONTHS,"absolute_survival_difference":CLINICALLY_RELEVANT_SURVIVAL_DIFFERENCE},"treatment":"Chemotherapy recorded vs not recorded among patients with radiotherapy recorded; concurrency cannot be established","bootstrap":"Patient-level nonparametric resampling; cohort, propensity model, weights, weighted adjusted Cox model, counterfactual curves, RMST, and contrasts are recalculated in every iteration","unavailable_fields":["surgery","postoperative radiotherapy","M stage","diagnosis year","separate treatment dates","chemotherapy agent/dose","radiation dose","margin status","ENE","PNI","performance status","major vs minor gland site"]}
    (out/"analysis_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(result.to_string(index=False)); print(f"Outputs: {out}")


if __name__ == "__main__": main()
