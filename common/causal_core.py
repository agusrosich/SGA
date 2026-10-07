"""Shared censoring-aware causal survival engine for the SEER-derived SGA cohort.

Used by both 01_entrenamiento_seer/train_seer_model.py (the broad T3/T4
training cohort) and 02_simulacion_rtog1008/simulate_rtog1008.py (the
RTOG-1008-like high-risk cohort). Cohort selection (load_data) lives in each
of those scripts, not here, because the two parts use different eligibility
masks over the same source extract.
"""

from __future__ import annotations

import warnings

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

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


def prepare_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive the modeling columns shared by every cohort definition."""
    d = raw.copy()
    d["Treatment"] = d["Chemotherapy_Binary"].astype(int)
    d["OS_Event"] = d["COD to site recode"].ne("Alive").astype(int)
    d["CSS_Event"] = d["COD to site recode"].eq("Salivary Gland").astype(int)
    d["Time"] = pd.to_numeric(d["Survival months"], errors="coerce").clip(lower=0)
    d["Time_120"] = d["Time"].clip(upper=TAU)
    d["OS_Event_120"] = ((d["OS_Event"] == 1) & (d["Time"] <= TAU)).astype(int)
    d["CSS_Event_120"] = ((d["CSS_Event"] == 1) & (d["Time"] <= TAU)).astype(int)
    d["T34"] = d["T_Unified"].astype(str).str.startswith(("T3", "T4"))
    d["N123"] = d["N_Unified"].isin(["N1", "N2", "N3"])
    d["Known_N"] = ~d["N_Unified"].isin(["NX", "N88"])
    d["Comparable_Histology"] = d["Histology_Unified"].isin(COMPARABLE_HISTOLOGIES)
    d["Squamous"] = d["Histology_Unified"].astype(str).str.contains("squamous", case=False)
    return d


def flow_step(flow: list, step: str, mask: pd.Series, previous_n: int) -> int:
    """Append one cohort-selection step and return the new remaining N."""
    remaining = int(mask.sum())
    flow.append({"Step": step, "Remaining_N": remaining, "Excluded_at_step_N": previous_n - remaining})
    return remaining


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


def bootstrap_wald_p(estimate_value: float, boot: pd.DataFrame, column: str) -> float:
    se = boot[column].std(ddof=1)
    return float(2 * stats.norm.sf(abs(estimate_value / se)))


PLOT_TEXT = {
    "en": {"rt": "RT only", "crt": "RT + chemotherapy", "months": "Months",
           "survival": "Adjusted marginal survival probability", "ate": "ATE: 10-year RMST difference (months)",
           "iterations": "Bootstrap iterations", "before": "Before weighting", "after": "After IPTW",
           "smd": "Standardized mean difference", "ps": "Propensity score", "density": "Density"},
    "es": {"rt": "Solo RT", "crt": "RT + quimioterapia", "months": "Meses",
           "survival": "Probabilidad de supervivencia marginal ajustada", "ate": "ATE: diferencia de RMST a 10 años (meses)",
           "iterations": "Iteraciones bootstrap", "before": "Antes de ponderar", "after": "Después de IPTW",
           "smd": "Diferencia de medias estandarizada", "ps": "Puntuación de propensión", "density": "Densidad"},
}
# Spanish figure labels, keyed by the English label that covariate_label() returns.
COVARIATE_LABELS_ES = {
    "Age": "Edad", "Male sex": "Sexo masculino", "T4nos": "T4 SAI",
    "Mucoepidermoid carcinoma": "Carcinoma mucoepidermoide", "Adenoid cystic carcinoma": "Carcinoma adenoide quístico",
    "Squamous cell carcinoma": "Carcinoma de células escamosas", "Carcinoma NOS": "Carcinoma SAI",
    "Small cell carcinoma": "Carcinoma de células pequeñas", "Adenocarcinoma": "Adenocarcinoma",
    "Unspecified neoplasms": "Neoplasias no especificadas", "Myoepithelial carcinoma": "Carcinoma mioepitelial",
    "Complex mixed and stromal neoplasms": "Neoplasias mixtas complejas y estromales",
    "Cystic, mucinous and serous neoplasms": "Neoplasias quísticas, mucinosas y serosas",
    "Squamous cell neoplasms": "Neoplasias de células escamosas", "Mixed tumor": "Tumor mixto",
    "Basal cell carcinoma": "Carcinoma basocelular", "Synovial-like neoplasms": "Neoplasias de tipo sinovial",
    "Adnexal and skin appendage neoplasms": "Neoplasias anexiales y de apéndices cutáneos",
    "Undifferentiated carcinoma": "Carcinoma indiferenciado", "Adenosquamous carcinoma": "Carcinoma adenoescamoso",
    "Sarcoma": "Sarcoma", "Adenomas and adenocarcinomas": "Adenomas y adenocarcinomas",
    "Carcinoma ex pleomorphic adenoma": "Carcinoma ex adenoma pleomorfo", "Oncocytic carcinoma": "Carcinoma oncocítico",
    "Epithelial neoplasms, NOS": "Neoplasias epiteliales SAI",
    "Epithelial-myoepithelial carcinoma": "Carcinoma epitelial-mioepitelial",
    "Mucinous adenocarcinoma": "Adenocarcinoma mucinoso", "Neuroendocrine carcinoma": "Carcinoma neuroendocrino",
}


def covariate_label(name: str, lang: str = "en") -> str:
    """Readable figure label for an encoded covariate (e.g. 'N_Unified_N2' -> 'N2')."""
    label = name
    if name == "Age_Median": label = "Age"
    elif name == "Sex_Male": label = "Male sex"
    elif name.startswith(("N_Unified_", "T_Unified_")):
        v = name[len("N_Unified_"):]; label = v[:2] + v[2:].lower()
    elif name.startswith("Histology_Unified_"):
        v = name[len("Histology_Unified_"):]; label = v[:1].upper() + v[1:]
    return COVARIATE_LABELS_ES.get(label, label) if lang == "es" else label


class _DecimalCommaFormatter(matplotlib.ticker.ScalarFormatter):
    """Tick labels with a decimal comma, for the Spanish figures."""
    def __call__(self, x, pos=None): return super().__call__(x, pos).replace(".", ",")


def plots(out, e: dict, boot: pd.DataFrame, smd: pd.DataFrame, dpi: int = 300, formats: tuple = ("png",), lang: str = "en") -> None:
    out.mkdir(parents=True,exist_ok=True); plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42})
    t=PLOT_TEXT[lang]
    def save(fig, stem):
        if lang == "es":
            for ax in fig.axes:
                for axis in (ax.xaxis, ax.yaxis):
                    if isinstance(axis.get_major_formatter(), matplotlib.ticker.ScalarFormatter): axis.set_major_formatter(_DecimalCommaFormatter())
        for fmt in formats: fig.savefig(out/f"{stem}.{fmt}",dpi=dpi)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for arm,label,color in [(0,t["rt"],"#475569"),(1,t["crt"],"#2563eb")]:
        z=e["curves"].query("Treatment==@arm"); ax.plot(z.Month,z.Survival,label=label,color=color,lw=2.2)
    ax.set(xlabel=t["months"], ylabel=t["survival"], ylim=(0,1.02), xlim=(0,TAU)); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); save(fig,"adjusted_marginal_survival")
    fig,ax=plt.subplots(figsize=(7,4.5)); z=boot.Delta_RMST_Months.dropna(); ax.hist(z,bins=30,color="#0f766e",alpha=.85); ax.axvline(0,color="black",ls="--"); ax.set(xlabel=t["ate"],ylabel=t["iterations"]); ax.grid(axis="y",alpha=.2); fig.tight_layout(); save(fig,"bootstrap_rmst_difference")
    top=smd.assign(abs_before=smd.SMD_Before.abs()).nlargest(20,"abs_before").sort_values("abs_before")
    labels=top.Covariate.map(lambda c: covariate_label(c, lang))
    fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(top.SMD_Before,labels,label=t["before"],color="#dc2626"); ax.scatter(top.SMD_After_IPTW,labels,label=t["after"],color="#2563eb"); ax.axvline(.1,color="grey",ls="--"); ax.axvline(-.1,color="grey",ls="--"); ax.set_xlabel(t["smd"]); ax.legend(loc="lower center",bbox_to_anchor=(0.5,1.0),ncol=2,frameon=False); fig.tight_layout(); save(fig,"covariate_balance_love_plot")
    fig,ax=plt.subplots(figsize=(7,4.5));
    for arm,label,color in [(0,t["rt"],"#475569"),(1,t["crt"],"#2563eb")]:
        ax.hist(e["data"].loc[e["data"].Treatment.eq(arm),"Propensity"],bins=25,density=True,alpha=.45,label=label,color=color)
    ax.set(xlabel=t["ps"],ylabel=t["density"]); ax.legend(); fig.tight_layout(); save(fig,"propensity_overlap")
