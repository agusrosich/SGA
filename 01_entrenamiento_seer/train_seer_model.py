"""Part 1: censoring-aware causal survival training on the broad SEER cohort.

Primary estimand: ATE in 10-year RMST (chemotherapy recorded vs no chemotherapy)
among T3/T4 patients with recorded radiotherapy. SEER fields in this extract do
not establish surgery, postoperative intent, concurrent timing, drug, dose, M
stage, grade, margins, ENE, PNI, performance status, diagnosis year, or gland
site. This is the general training/estimation pass over the full eligible
population; see 02_simulacion_rtog1008/simulate_rtog1008.py for the
RTOG-1008-like high-risk cohort built on top of these results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import common.causal_core as core


def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    d = core.prepare_frame(raw)
    flow: list = []
    mask = pd.Series(True, index=d.index)
    n = core.flow_step(flow, "Source standardized records", mask, len(d))
    mask &= d["T34"]; n = core.flow_step(flow, "T3/T4 disease", mask, n)
    mask &= d["Known_N"]; n = core.flow_step(flow, "Known N category", mask, n)
    mask &= d["Radiation_Binary"].eq(1); n = core.flow_step(flow, "Radiotherapy recorded", mask, n)
    mask &= d["Time"].notna(); n = core.flow_step(flow, "Nonmissing survival time", mask, n)
    return d.loc[mask].copy().reset_index(drop=True), pd.DataFrame(flow)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="01_entrenamiento_seer/data/raw/ExportadaSEER_Estandarizada.csv")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=core.SEED)
    ap.add_argument("--reuse-bootstrap", action="store_true", help="Reuse an existing complete bootstrap table while regenerating other outputs.")
    args = ap.parse_args()

    out = ROOT / "01_entrenamiento_seer" / "outputs"
    tables = out / "tables"; figs = out / "figures"; tables.mkdir(parents=True, exist_ok=True)

    d, flow = load_data(ROOT / args.input)
    e = core.estimate(d)

    bootstrap_path = tables / "bootstrap.csv"
    if args.reuse_bootstrap and bootstrap_path.exists():
        boot = pd.read_csv(bootstrap_path)
    else:
        boot = core.bootstrap(d, args.bootstrap, args.seed)

    smd = core.smd_table(e["data"])
    valid = core.validation(d, 5, args.seed)
    sens = core.sensitivity(d)
    ci = np.nanpercentile(boot.Delta_RMST_Months, [2.5, 97.5])
    delta5 = e["s60_1"] - e["s60_0"]; delta10 = e["s120_1"] - e["s120_0"]

    result = pd.DataFrame([{
        "Estimand": "ATE: difference in 10-year RMST",
        "Population": "T3/T4, known N, radiotherapy recorded",
        "N": len(d), "Treated_N": int(d.Treatment.sum()), "Control_N": int((1 - d.Treatment).sum()),
        "RMST_No_Chemotherapy_Months": e["rmst0"], "RMST_Chemotherapy_Months": e["rmst1"],
        "Delta_RMST_Months": e["delta_rmst"], "CI95_Lower": ci[0], "CI95_Upper": ci[1],
        "P_Value_RMST": core.bootstrap_wald_p(e["delta_rmst"], boot, "Delta_RMST_Months"),
        "Survival_Difference_5y": delta5, "P_Value_Survival_5y": core.bootstrap_wald_p(delta5, boot, "Survival_Difference_5y"),
        "Survival_Difference_10y": delta10, "P_Value_Survival_10y": core.bootstrap_wald_p(delta10, boot, "Survival_Difference_10y"),
        "Reverse_KM_Median_Followup_Months": core.reverse_km_followup(d),
        "Bootstrap_Successful": int(boot.Delta_RMST_Months.notna().sum()),
    }])

    missing = pd.DataFrame({"Variable": d.columns, "Missing_N": [d[c].isna().sum() for c in d.columns],
                             "Missing_Percent": [d[c].isna().mean() * 100 for c in d.columns]})
    ps1 = e["data"].query("Treatment==1").Propensity; ps0 = e["data"].query("Treatment==0").Propensity
    common_lo = max(ps1.quantile(.01), ps0.quantile(.01)); common_hi = min(ps1.quantile(.99), ps0.quantile(.99))
    positivity = pd.DataFrame([
        {"Group": "RT + chemotherapy", "N": len(ps1), "PS_Min": ps1.min(), "PS_P1": ps1.quantile(.01), "PS_P99": ps1.quantile(.99), "PS_Max": ps1.max(), "Percent_in_1_99_Common_Support": ps1.between(common_lo, common_hi).mean() * 100},
        {"Group": "RT only", "N": len(ps0), "PS_Min": ps0.min(), "PS_P1": ps0.quantile(.01), "PS_P99": ps0.quantile(.99), "PS_Max": ps0.max(), "Percent_in_1_99_Common_Support": ps0.between(common_lo, common_hi).mean() * 100},
    ])
    css = core.estimate(d, event="CSS_Event_120")
    css_result = pd.DataFrame([{"Endpoint": "Cancer-specific mortality (separate secondary cause-specific model)",
                                 "Delta_RMST_Months": css["delta_rmst"], "Survival_Difference_5y": css["s60_1"] - css["s60_0"],
                                 "Survival_Difference_10y": css["s120_1"] - css["s120_0"]}])

    patient_level_predictions = pd.concat([
        e["data"][["Age_Median", "Sex", "T_Unified", "N_Unified", "Histology_Unified", "Treatment"]].reset_index(drop=True),
        e["individual"].reset_index(drop=True),
    ], axis=1)

    outputs = {
        "primary_result": result, "cohort_flow": flow, "baseline_observed_groups": core.baseline_table(d),
        "available_treatment_timing": core.available_treatment_timing(d), "balance": smd, "positivity": positivity,
        "bootstrap": boot, "validation": valid, "sensitivity": sens,
        "histology_sensitivity": core.histology_sensitivity(d), "unmeasured_confounding_evalue": core.evalue_table(e["cox"]),
        "missingness": missing, "cancer_specific_secondary": css_result, "adjusted_survival_curves": e["curves"],
        "patient_level_predictions": patient_level_predictions,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / f"{name}.csv", index=False)
    core.plots(figs, e, boot, smd)

    metadata = {
        "part": "1 of 2 - SEER training on the broad T3/T4 eligible cohort",
        "primary_endpoint": "Overall survival with living patients censored at last follow-up",
        "estimand": "ATE difference in RMST through 120 months",
        "prespecified_clinical_relevance": {"absolute_RMST_months": core.CLINICALLY_RELEVANT_RMST_MONTHS, "absolute_survival_difference": core.CLINICALLY_RELEVANT_SURVIVAL_DIFFERENCE},
        "treatment": "Chemotherapy recorded vs not recorded among patients with radiotherapy recorded; concurrency cannot be established",
        "bootstrap": "Patient-level nonparametric resampling; cohort, propensity model, weights, weighted adjusted Cox model, counterfactual curves, RMST, and contrasts are recalculated in every iteration",
        "unavailable_fields": ["surgery", "postoperative radiotherapy", "M stage", "diagnosis year", "separate treatment dates", "chemotherapy agent/dose", "radiation dose", "margin status", "ENE", "PNI", "performance status", "major vs minor gland site"],
        "see_also": "02_simulacion_rtog1008/simulate_rtog1008.py builds the RTOG-1008-like high-risk cohort and simulated-trial comparison on top of this analysis",
    }
    (out / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(result.to_string(index=False)); print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
