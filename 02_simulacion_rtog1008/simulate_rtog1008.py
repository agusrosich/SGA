"""Part 2: RTOG-1008-like simulation, as an explicit pre-results prediction.

RTOG 1008's actual eligibility (NCT01220583) is pathologic T3-4 OR N1-3 OR
(T1-2,N0 with a close [<=1mm] or microscopically positive margin), restricted
to a short list of histologies, resected with curative intent, M0. SEER does
not record surgery, margins, or grade in this extract, so the margin-based
T1-2N0 branch cannot be applied and is declared as not verifiable rather than
approximated. What SEER *can* support directly is the T3-4 OR N1-3 branch
(both T and N are recorded) plus a histology restriction to the families
comparable with the RTOG 1008 list.

This is therefore a distinct, broader-in-N and narrower-in-histology cohort
than the one used in 01_entrenamiento_seer/train_seer_model.py, and its
propensity/outcome model is refit from scratch on this cohort rather than
reusing Part 1's fitted model. On top of the standardized ATE for this
cohort, this script repeats the RTOG 1008 trial design (1:1 random
assignment, N=252) many times over the estimated individual potential
outcomes, framed explicitly as this analysis's falsifiable, pre-results
prediction for what RTOG 1008's real result will look like.
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
    mask &= (d["T34"] | d["N123"])
    n = core.flow_step(flow, "T3-4 OR N1-3 (RTOG 1008 high-risk stage; T1-2N0-with-margin branch not verifiable in SEER)", mask, n)
    mask &= d["Known_N"]; n = core.flow_step(flow, "Known N category", mask, n)
    mask &= d["Radiation_Binary"].eq(1); n = core.flow_step(flow, "Radiotherapy recorded", mask, n)
    mask &= d["Time"].notna(); n = core.flow_step(flow, "Nonmissing survival time", mask, n)
    mask &= d["Comparable_Histology"]; n = core.flow_step(flow, "Histology comparable with RTOG 1008", mask, n)
    return d.loc[mask].copy().reset_index(drop=True), pd.DataFrame(flow)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="01_entrenamiento_seer/data/raw/ExportadaSEER_Estandarizada.csv")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--simulations", type=int, default=1000)
    ap.add_argument("--trial-size", type=int, default=252, help="RTOG 1008's actual Phase III enrollment target.")
    ap.add_argument("--seed", type=int, default=core.SEED)
    ap.add_argument("--reuse-bootstrap", action="store_true")
    args = ap.parse_args()

    out = ROOT / "02_simulacion_rtog1008" / "outputs"
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
    ci = np.nanpercentile(boot.Delta_RMST_Months, [2.5, 97.5])
    delta5 = e["s60_1"] - e["s60_0"]; delta10 = e["s120_1"] - e["s120_0"]

    result = pd.DataFrame([{
        "Estimand": "ATE: difference in 10-year RMST",
        "Population": "RTOG 1008-like high-risk cohort (T3-4 OR N1-3, known N, radiotherapy recorded, comparable histology)",
        "N": len(d), "Treated_N": int(d.Treatment.sum()), "Control_N": int((1 - d.Treatment).sum()),
        "RMST_No_Chemotherapy_Months": e["rmst0"], "RMST_Chemotherapy_Months": e["rmst1"],
        "Delta_RMST_Months": e["delta_rmst"], "CI95_Lower": ci[0], "CI95_Upper": ci[1],
        "P_Value_RMST": core.bootstrap_wald_p(e["delta_rmst"], boot, "Delta_RMST_Months"),
        "Survival_Difference_5y": delta5, "P_Value_Survival_5y": core.bootstrap_wald_p(delta5, boot, "Survival_Difference_5y"),
        "Survival_Difference_10y": delta10, "P_Value_Survival_10y": core.bootstrap_wald_p(delta10, boot, "Survival_Difference_10y"),
        "Reverse_KM_Median_Followup_Months": core.reverse_km_followup(d),
        "Bootstrap_Successful": int(boot.Delta_RMST_Months.notna().sum()),
    }])

    ps1 = e["data"].query("Treatment==1").Propensity; ps0 = e["data"].query("Treatment==0").Propensity
    common_lo = max(ps1.quantile(.01), ps0.quantile(.01)); common_hi = min(ps1.quantile(.99), ps0.quantile(.99))
    positivity = pd.DataFrame([
        {"Group": "RT + chemotherapy", "N": len(ps1), "PS_Min": ps1.min(), "PS_P1": ps1.quantile(.01), "PS_P99": ps1.quantile(.99), "PS_Max": ps1.max(), "Percent_in_1_99_Common_Support": ps1.between(common_lo, common_hi).mean() * 100},
        {"Group": "RT only", "N": len(ps0), "PS_Min": ps0.min(), "PS_P1": ps0.quantile(.01), "PS_P99": ps0.quantile(.99), "PS_Max": ps0.max(), "Percent_in_1_99_Common_Support": ps0.between(common_lo, common_hi).mean() * 100},
    ])

    sims = core.simulated_trials(e["data"], e["individual"], args.simulations, args.seed, n=args.trial_size)

    outputs = {
        "primary_result": result, "cohort_flow": flow, "baseline_observed_groups": core.baseline_table(d),
        "balance": smd, "positivity": positivity, "bootstrap": boot, "validation": valid,
        f"simulated_trials_{args.trial_size}": sims,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / f"{name}.csv", index=False)
    core.plots(figs, e, boot, smd)

    metadata = {
        "part": "2 of 2 - RTOG 1008-like high-risk cohort and simulated-trial prediction",
        "eligibility_source": "NCT01220583 (RTOG 1008): pathologic T3-4 OR N1-3 OR (T1-2,N0 with close/positive margin)",
        "eligibility_applied": "T3-4 OR N1-3, known N, radiotherapy recorded, histology comparable with RTOG 1008 list",
        "eligibility_not_verifiable": "T1-2,N0-with-margin branch: SEER does not record surgical margin status",
        "purpose": "Falsifiable pre-results prediction for RTOG 1008's real Phase III result, not a substitute for it",
        "simulated_trial_design": f"{args.simulations} repetitions of a 1:1 random-assignment trial of N={args.trial_size} (RTOG 1008's actual Phase III enrollment), drawn from this cohort's estimated individual potential outcomes",
    }
    (out / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(result.to_string(index=False)); print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
