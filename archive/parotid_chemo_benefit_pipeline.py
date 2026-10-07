"""
Predictive chemotherapy-benefit pipeline for advanced parotid/salivary gland cancer.

Primary question:
Does adding chemotherapy to radiotherapy improve survival in T3/T4 disease?

The script builds publication-ready English tables and an in-silico randomized trial
modeled after the RTOG 1008 design: radiation alone vs concurrent chemoradiation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
MAX_FOLLOW_UP_MONTHS = 120


def assign_stage(row: pd.Series) -> str:
    t = str(row["T_Unified"]).upper()
    n = str(row["N_Unified"]).upper()

    if any(prefix in t for prefix in ["TX", "T88", "T0"]) or any(prefix in n for prefix in ["NX", "N88"]):
        return "Unknown"
    if t.startswith("T1") and n.startswith("N0"):
        return "Stage I"
    if (t.startswith("T2") and n.startswith("N0")) or (t.startswith("T1") and n.startswith("N1")):
        return "Stage II"
    if (t.startswith("T3") and n.startswith("N0")) or (t.startswith(("T1", "T2")) and n.startswith(("N1", "N2"))):
        return "Stage III"
    if t.startswith("T4") or n.startswith(("N2", "N3")) or (t.startswith("T3") and n.startswith(("N2", "N3"))):
        return "Stage IV"
    return "Stage III"


def t_group(value: object) -> str:
    t = str(value).upper()
    if t.startswith("T3") or t.startswith("T4"):
        return "T3/T4"
    if t.startswith("T1") or t.startswith("T2"):
        return "T1/T2"
    return "Unknown"


def read_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Cancer_Death_Event"] = (df["COD to site recode"] != "Alive").astype(int)
    df["Overall_Survival_Months"] = df["Survival months"].clip(upper=MAX_FOLLOW_UP_MONTHS)
    df["Cancer_Death_Event_Capped"] = np.where(
        df["Survival months"] > MAX_FOLLOW_UP_MONTHS,
        0,
        df["Cancer_Death_Event"],
    )
    df["Stage_Group"] = df.apply(assign_stage, axis=1)
    df["T_Group"] = df["T_Unified"].apply(t_group)
    df["Chemoradiation"] = ((df["Radiation_Binary"] == 1) & (df["Chemotherapy_Binary"] == 1)).astype(int)
    df["Radiation_Alone"] = ((df["Radiation_Binary"] == 1) & (df["Chemotherapy_Binary"] == 0)).astype(int)
    df["Treatment_Arm_Observed"] = np.select(
        [
            df["Chemoradiation"] == 1,
            df["Radiation_Alone"] == 1,
            (df["Radiation_Binary"] == 0) & (df["Chemotherapy_Binary"] == 1),
        ],
        ["Chemoradiation", "Radiation alone", "Chemotherapy alone"],
        default="No radiation or chemotherapy",
    )
    return df[df["Stage_Group"] != "Unknown"].copy()


def make_preprocessor(categorical: Iterable[str], numeric: Iterable[str]) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), list(numeric)),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), list(categorical)),
        ],
        remainder="drop",
    )


def propensity_score_match(df: pd.DataFrame, covariates: list[str], treatment_col: str = "Chemoradiation") -> pd.DataFrame:
    model_df = df.dropna(subset=covariates + [treatment_col]).copy()
    if model_df[treatment_col].nunique() < 2:
        return pd.DataFrame()

    categorical = [c for c in covariates if model_df[c].dtype == "object"]
    numeric = [c for c in covariates if c not in categorical]
    preprocessor = make_preprocessor(categorical, numeric)
    ps_model = Pipeline(
        [
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )
    ps_model.fit(model_df[covariates], model_df[treatment_col])
    model_df["Propensity_Score"] = ps_model.predict_proba(model_df[covariates])[:, 1]

    treated = model_df[model_df[treatment_col] == 1].copy()
    control = model_df[model_df[treatment_col] == 0].copy()
    if treated.empty or control.empty:
        return pd.DataFrame()

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[["Propensity_Score"]])
    _, indices = nn.kneighbors(treated[["Propensity_Score"]])
    matched_control = control.iloc[indices.ravel()].copy()
    matched_control = matched_control.loc[~matched_control.index.duplicated(keep="first")]
    matched_treated = treated.iloc[: len(matched_control)].copy()
    return pd.concat([matched_treated, matched_control], ignore_index=True)


def survival_summary(df: pd.DataFrame, label: str) -> dict[str, object]:
    arms = {
        "Chemoradiation": df[df["Chemoradiation"] == 1],
        "Radiation alone": df[df["Chemoradiation"] == 0],
    }
    rows: dict[str, object] = {"Analysis_Set": label}
    for arm_name, arm_df in arms.items():
        rows[f"{arm_name}_N"] = len(arm_df)
        rows[f"{arm_name}_Median_OS_Months"] = arm_df["Overall_Survival_Months"].median()
        rows[f"{arm_name}_Cancer_Mortality_Percent"] = arm_df["Cancer_Death_Event_Capped"].mean() * 100

    if min(len(arms["Chemoradiation"]), len(arms["Radiation alone"])) >= 5:
        lr = logrank_test(
            arms["Chemoradiation"]["Overall_Survival_Months"],
            arms["Radiation alone"]["Overall_Survival_Months"],
            arms["Chemoradiation"]["Cancer_Death_Event_Capped"],
            arms["Radiation alone"]["Cancer_Death_Event_Capped"],
        )
        rows["Logrank_P_Value"] = lr.p_value
    else:
        rows["Logrank_P_Value"] = np.nan

    rows["Median_OS_Difference_Months"] = (
        rows["Chemoradiation_Median_OS_Months"] - rows["Radiation alone_Median_OS_Months"]
    )
    rows["Mortality_Reduction_Percent"] = (
        rows["Radiation alone_Cancer_Mortality_Percent"] - rows["Chemoradiation_Cancer_Mortality_Percent"]
    )
    return rows


def train_outcome_model(df: pd.DataFrame, features: list[str], categorical: list[str], numeric: list[str]) -> tuple[Pipeline, pd.DataFrame]:
    X = df[features]
    y = df["Cancer_Death_Event_Capped"]
    classifier = Pipeline(
        [
            ("prep", make_preprocessor(categorical, numeric)),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = cross_val_score(classifier, X, y, cv=cv, scoring="roc_auc")
    classifier.fit(X, y)
    apparent_auc = roc_auc_score(y, classifier.predict_proba(X)[:, 1])
    metrics = pd.DataFrame(
        [
            {
                "Model": "Gradient boosting mortality classifier",
                "Endpoint": "10-year capped cancer-specific mortality",
                "Apparent_AUC": apparent_auc,
                "Cross_Validated_AUC_Mean": auc_scores.mean(),
                "Cross_Validated_AUC_SD": auc_scores.std(),
                "N": len(df),
            }
        ]
    )
    return classifier, metrics


def estimate_individual_benefit(df: pd.DataFrame, features: list[str], categorical: list[str], numeric: list[str]) -> pd.DataFrame:
    treated = df[df["Chemoradiation"] == 1].copy()
    control = df[df["Chemoradiation"] == 0].copy()
    if min(len(treated), len(control)) < 20:
        raise ValueError("Not enough treated/control patients to estimate counterfactual benefit.")

    base_steps = [
        ("prep", make_preprocessor(categorical, numeric)),
        ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=8, random_state=RANDOM_STATE)),
    ]
    treated_model = Pipeline(base_steps)
    control_model = Pipeline(
        [
            ("prep", make_preprocessor(categorical, numeric)),
            ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=8, random_state=RANDOM_STATE + 1)),
        ]
    )
    treated_model.fit(treated[features], treated["Overall_Survival_Months"])
    control_model.fit(control[features], control["Overall_Survival_Months"])

    out = df.copy()
    out["Predicted_OS_With_ChemoRT_Months"] = treated_model.predict(out[features])
    out["Predicted_OS_With_RT_Alone_Months"] = control_model.predict(out[features])
    out["Predicted_Chemo_Benefit_Months"] = (
        out["Predicted_OS_With_ChemoRT_Months"] - out["Predicted_OS_With_RT_Alone_Months"]
    )
    out["Predicted_Benefit_Group"] = pd.cut(
        out["Predicted_Chemo_Benefit_Months"],
        bins=[-np.inf, 0, 6, 12, np.inf],
        labels=["No predicted benefit", "Small benefit", "Moderate benefit", "Large benefit"],
    )
    return out


def subgroup_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subgroup_specs = [
        ("T stage", "T_Unified"),
        ("N stage", "N_Unified"),
        ("Stage group", "Stage_Group"),
        ("Histology", "Histology_Unified"),
        ("Predicted benefit group", "Predicted_Benefit_Group"),
    ]
    for group_label, col in subgroup_specs:
        for value, subset in df.groupby(col, dropna=False, observed=False):
            if len(subset) < 20:
                continue
            rows.append(
                {
                    "Subgroup_Type": group_label,
                    "Subgroup": value,
                    "N": len(subset),
                    "Observed_Chemoradiation_Rate_Percent": subset["Chemoradiation"].mean() * 100,
                    "Mean_Predicted_Benefit_Months": subset["Predicted_Chemo_Benefit_Months"].mean(),
                    "Median_Predicted_Benefit_Months": subset["Predicted_Chemo_Benefit_Months"].median(),
                    "Patients_With_Positive_Predicted_Benefit_Percent": (subset["Predicted_Chemo_Benefit_Months"] > 0).mean() * 100,
                }
            )
    return pd.DataFrame(rows).sort_values("Mean_Predicted_Benefit_Months", ascending=False)


def simulate_trial(
    df: pd.DataFrame,
    trial_n: int,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    eligible = df.copy()
    if len(eligible) < trial_n:
        replace = True
    else:
        replace = False

    trial = eligible.sample(n=trial_n, replace=replace, random_state=seed).reset_index(drop=True)
    assignments = np.array(["Chemoradiation"] * (trial_n // 2) + ["Radiation alone"] * (trial_n - trial_n // 2))
    rng.shuffle(assignments)
    trial["Simulated_Arm"] = assignments
    trial["Simulated_OS_Months"] = np.where(
        trial["Simulated_Arm"] == "Chemoradiation",
        trial["Predicted_OS_With_ChemoRT_Months"],
        trial["Predicted_OS_With_RT_Alone_Months"],
    )
    trial["Simulated_Cancer_Death_Event"] = np.where(trial["Simulated_OS_Months"] >= MAX_FOLLOW_UP_MONTHS, 0, 1)

    rt = trial[trial["Simulated_Arm"] == "Radiation alone"]
    crt = trial[trial["Simulated_Arm"] == "Chemoradiation"]
    lr = logrank_test(
        crt["Simulated_OS_Months"],
        rt["Simulated_OS_Months"],
        crt["Simulated_Cancer_Death_Event"],
        rt["Simulated_Cancer_Death_Event"],
    )

    bootstrap_effects = []
    for i in range(bootstrap_iterations):
        boot = eligible.sample(n=trial_n, replace=True, random_state=seed + i + 1).reset_index(drop=True)
        assignments = np.array(["Chemoradiation"] * (trial_n // 2) + ["Radiation alone"] * (trial_n - trial_n // 2))
        rng.shuffle(assignments)
        boot["Simulated_Arm"] = assignments
        os_months = np.where(
            boot["Simulated_Arm"] == "Chemoradiation",
            boot["Predicted_OS_With_ChemoRT_Months"],
            boot["Predicted_OS_With_RT_Alone_Months"],
        )
        boot["Simulated_OS_Months"] = os_months
        delta = (
            boot.loc[boot["Simulated_Arm"] == "Chemoradiation", "Simulated_OS_Months"].median()
            - boot.loc[boot["Simulated_Arm"] == "Radiation alone", "Simulated_OS_Months"].median()
        )
        bootstrap_effects.append({"Iteration": i + 1, "Delta_OS_Months": delta})

    bootstrap_df = pd.DataFrame(bootstrap_effects)
    effect = bootstrap_df["Delta_OS_Months"].to_numpy()
    result = pd.DataFrame(
        [
            {
                "Design": "RTOG 1008-like in-silico randomized trial",
                "Trial_N": trial_n,
                "Chemoradiation_N": len(crt),
                "Radiation_Alone_N": len(rt),
                "Median_OS_Chemoradiation_Months": crt["Simulated_OS_Months"].median(),
                "Median_OS_Radiation_Alone_Months": rt["Simulated_OS_Months"].median(),
                "Median_OS_Difference_Months": crt["Simulated_OS_Months"].median() - rt["Simulated_OS_Months"].median(),
                "Logrank_P_Value": lr.p_value,
                "Bootstrap_Iterations": bootstrap_iterations,
                "Bootstrap_Delta_OS_Mean": effect.mean(),
                "Bootstrap_Delta_OS_95CI_Lower": np.percentile(effect, 2.5),
                "Bootstrap_Delta_OS_95CI_Upper": np.percentile(effect, 97.5),
                "Probability_Delta_OS_Greater_Than_0": (effect > 0).mean(),
            }
        ]
    )
    return trial, result, bootstrap_df


def write_summary(
    path: Path,
    cohort: pd.DataFrame,
    observed: pd.DataFrame,
    matched: pd.DataFrame,
    model_metrics: pd.DataFrame,
    trial_results: pd.DataFrame,
) -> None:
    lines = [
        "Predictive Chemotherapy Benefit Analysis",
        "========================================",
        "",
        "Clinical question: Does adding chemotherapy to radiotherapy improve survival in T3/T4 disease?",
        "Target comparison: concurrent chemoradiation vs radiation alone.",
        "Trial simulation: RTOG 1008-like equal-arm in-silico randomization.",
        "",
        f"Eligible analysis cohort: {len(cohort):,} T3/T4 patients treated with radiation alone or chemoradiation.",
        "",
        "Observed comparison:",
        observed.to_string(index=False),
        "",
        "Propensity-matched observed comparison:",
        matched.to_string(index=False),
        "",
        "Predictive model performance:",
        model_metrics.to_string(index=False),
        "",
        "In-silico trial result:",
        trial_results.to_string(index=False),
        "",
        "Interpretation note:",
        "This is a retrospective predictive simulation, not causal proof. Results should be reported as hypothesis-generating",
        "and require external validation or prospective confirmation.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_n_pct(count: int, denominator: int) -> str:
    pct = (count / denominator * 100) if denominator else 0
    return f"{count} ({pct:.1f}%)"


def _format_median_iqr(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "NA"
    q1, q3 = clean.quantile([0.25, 0.75])
    return f"{clean.median():.1f} ({q1:.1f}-{q3:.1f})"


def make_simulated_population_table(trial_patients: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("Overall", trial_patients),
        ("Chemoradiation", trial_patients[trial_patients["Simulated_Arm"] == "Chemoradiation"]),
        ("Radiation alone", trial_patients[trial_patients["Simulated_Arm"] == "Radiation alone"]),
    ]
    rows = []

    rows.append({"Characteristic": "N", **{name: str(len(group)) for name, group in groups}})
    rows.append(
        {
            "Characteristic": "Age, median (IQR), years",
            **{name: _format_median_iqr(group["Age_Median"]) for name, group in groups},
        }
    )

    categorical_specs = [
        ("Sex", "Sex"),
        ("T stage", "T_Unified"),
        ("N stage", "N_Unified"),
        ("Stage group", "Stage_Group"),
        ("Histology", "Histology_Unified"),
        ("Predicted chemotherapy benefit group", "Predicted_Benefit_Group"),
    ]
    for label, column in categorical_specs:
        levels = list(trial_patients[column].dropna().astype(str).value_counts().index)
        for level in levels:
            row = {"Characteristic": f"{label}: {level}"}
            for name, group in groups:
                count = (group[column].astype(str) == level).sum()
                row[name] = _format_n_pct(int(count), len(group))
            rows.append(row)

    return pd.DataFrame(rows)


def write_trial_figures(output_dir: Path, trial_patients: pd.DataFrame, trial_results: pd.DataFrame, bootstrap_df: pd.DataFrame) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#64748b", "#2563eb"]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    kmf = KaplanMeierFitter()
    for arm, color in zip(["Radiation alone", "Chemoradiation"], colors):
        subset = trial_patients[trial_patients["Simulated_Arm"] == arm]
        kmf.fit(
            subset["Simulated_OS_Months"],
            event_observed=subset["Simulated_Cancer_Death_Event"],
            label=f"{arm} (n={len(subset)})",
        )
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color, linewidth=2.2)
    ax.set_xlabel("Predicted overall survival (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title("Kaplan-Meier curves in the in-silico randomized trial")
    ax.set_xlim(left=0, right=MAX_FOLLOW_UP_MONTHS)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figures_dir / "kaplan_meier_predicted_os_by_arm.png", dpi=200)
    plt.close(fig)

    result = trial_results.iloc[0]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(bootstrap_df["Delta_OS_Months"], bins=30, color="#0f766e", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="No difference")
    ax.axvline(result["Median_OS_Difference_Months"], color="#dc2626", linewidth=1.6, label="Observed simulated delta")
    ax.set_xlabel("Delta median OS (chemoradiation - radiation alone), months")
    ax.set_ylabel("Bootstrap simulations")
    ax.set_title("Bootstrap uncertainty of treatment effect")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "bootstrap_delta_os_distribution.png", dpi=200)
    plt.close(fig)

    composition = (
        trial_patients.groupby(["Simulated_Arm", "Stage_Group"])
        .size()
        .reset_index(name="N")
    )
    pivot = composition.pivot(index="Simulated_Arm", columns="Stage_Group", values="N").fillna(0)
    pivot = pivot.reindex(["Radiation alone", "Chemoradiation"])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bottom = np.zeros(len(pivot))
    palette = ["#16a34a", "#eab308", "#f97316", "#b91c1c"]
    for idx, column in enumerate(pivot.columns):
        values = pivot[column].to_numpy()
        ax.bar(pivot.index, values, bottom=bottom, label=column, color=palette[idx % len(palette)])
        bottom += values
    ax.set_ylabel("Patients")
    ax.set_title("Simulated trial population by stage")
    ax.legend(title="Stage", loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "simulated_trial_stage_distribution.png", dpi=200)
    plt.close(fig)


def write_focused_in_silico_outputs(
    output_dir: Path,
    cohort: pd.DataFrame,
    trial_patients: pd.DataFrame,
    trial_results: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    model_metrics: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    arm_characteristics = (
        trial_patients.groupby("Simulated_Arm")
        .agg(
            N=("Age_Median", "size"),
            Age_Mean=("Age_Median", "mean"),
            T4_Percent=("T_Unified", lambda s: s.astype(str).str.startswith("T4").mean() * 100),
            Node_Positive_Percent=("N_Unified", lambda s: (~s.astype(str).eq("N0")).mean() * 100),
            Stage_IV_Percent=("Stage_Group", lambda s: s.astype(str).eq("Stage IV").mean() * 100),
            Predicted_OS_Median_Months=("Simulated_OS_Months", "median"),
            Predicted_OS_Mean_Months=("Simulated_OS_Months", "mean"),
            Predicted_Event_Percent=("Simulated_Cancer_Death_Event", lambda s: s.mean() * 100),
        )
        .reset_index()
    )

    primary = trial_results.copy()
    primary.insert(0, "Disease_Context", "Advanced T3/T4 salivary gland cancer")
    primary.insert(1, "Comparison", "Concurrent chemoradiation vs radiation alone")
    primary["Conclusion"] = np.where(
        (primary["Bootstrap_Delta_OS_95CI_Lower"] > 0) & (primary["Logrank_P_Value"] < 0.05),
        "In-silico trial predicts a statistically supported survival benefit from adding chemotherapy.",
        "In-silico trial does not predict a statistically robust overall survival benefit from adding chemotherapy.",
    )

    primary.to_csv(output_dir / "primary_in_silico_trial_result.csv", index=False)
    arm_characteristics.to_csv(output_dir / "simulated_trial_arm_characteristics.csv", index=False)
    make_simulated_population_table(trial_patients).to_csv(output_dir / "simulated_trial_population_characteristics.csv", index=False)
    bootstrap_df.to_csv(output_dir / "bootstrap_delta_os_distribution.csv", index=False)
    trial_patients.to_csv(output_dir / "simulated_trial_patient_level_predictions.csv", index=False)
    write_trial_figures(output_dir, trial_patients, trial_results, bootstrap_df)

    result = primary.iloc[0]
    model = model_metrics.iloc[0]
    payload = {
        "title": "Salivary Gland Cancer In-Silico Trial Results",
        "clinical_question": "Does adding chemotherapy to radiotherapy improve overall survival in T3/T4 salivary gland cancer?",
        "eligible_retrospective_cohort_n": int(len(cohort)),
        "primary_result": primary.to_dict(orient="records"),
        "simulated_arm_characteristics": arm_characteristics.to_dict(orient="records"),
        "model_performance": model_metrics.to_dict(orient="records"),
        "bootstrap_delta_os_distribution": bootstrap_df.to_dict(orient="records"),
    }
    (output_dir / "in_silico_salivary_gland_trial_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    report = f"""# Salivary Gland Cancer In-Silico Trial Results

## Clinical Question
Does adding chemotherapy to radiotherapy improve overall survival in patients with advanced T3/T4 salivary gland cancer?

## In-Silico Trial Design
- Design: RTOG 1008-like randomized in-silico trial.
- Population: T3/T4 salivary gland cancer patients treated with radiation alone or concurrent chemoradiation in the source cohort.
- Eligible retrospective cohort: {len(cohort):,} patients.
- Simulated sample size: {int(result['Trial_N']):,} patients.
- Randomization: 1:1 chemoradiation vs radiation alone.
- Endpoint: predicted overall survival in months, administratively capped at {MAX_FOLLOW_UP_MONTHS} months.
- Uncertainty: {int(result['Bootstrap_Iterations']):,} bootstrap simulations.

## Predictive Model
- Model: {model['Model']}.
- Endpoint used for model performance: {model['Endpoint']}.
- Apparent AUC: {model['Apparent_AUC']:.3f}.
- Cross-validated AUC: {model['Cross_Validated_AUC_Mean']:.3f} +/- {model['Cross_Validated_AUC_SD']:.3f}.

## Primary Simulated Trial Result
- Chemoradiation arm: {int(result['Chemoradiation_N']):,} patients.
- Radiation-alone arm: {int(result['Radiation_Alone_N']):,} patients.
- Median predicted OS with chemoradiation: {result['Median_OS_Chemoradiation_Months']:.1f} months.
- Median predicted OS with radiation alone: {result['Median_OS_Radiation_Alone_Months']:.1f} months.
- Median OS difference: {result['Median_OS_Difference_Months']:+.1f} months.
- Log-rank p-value: {result['Logrank_P_Value']:.4f}.
- Bootstrap mean OS difference: {result['Bootstrap_Delta_OS_Mean']:+.1f} months.
- Bootstrap 95% CI: {result['Bootstrap_Delta_OS_95CI_Lower']:+.1f} to {result['Bootstrap_Delta_OS_95CI_Upper']:+.1f} months.
- Probability that OS difference is greater than 0: {result['Probability_Delta_OS_Greater_Than_0']:.3f}.

## Interpretation
{result['Conclusion']}

This output is restricted to the in-silico salivary gland cancer trial. It should be interpreted as a retrospective predictive simulation and hypothesis-generating evidence, not as prospective causal proof.
"""
    (output_dir / "in_silico_trial_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run English publication tables and RTOG 1008-like in-silico simulation.")
    parser.add_argument("--input", default="data/raw/ExportadaSEER_Estandarizada.csv", help="Input standardized SEER CSV.")
    parser.add_argument("--trial-n", type=int, default=252, help="Simulated trial sample size. RTOG 1008 Phase III planned N=252.")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations for simulated treatment effect uncertainty.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_path = root / args.input
    tables_dir = root / "outputs" / "tables"
    reports_dir = root / "outputs" / "reports"
    focused_trial_dir = root / "outputs" / "in_silico_salivary_gland_trial"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = read_and_prepare(input_path)
    analysis = df[
        (df["T_Group"] == "T3/T4")
        & (df["Radiation_Binary"] == 1)
        & (df["Treatment_Arm_Observed"].isin(["Chemoradiation", "Radiation alone"]))
    ].copy()

    features = ["Age_Median", "Sex", "T_Unified", "N_Unified", "Stage_Group", "Histology_Unified"]
    categorical = ["Sex", "T_Unified", "N_Unified", "Stage_Group", "Histology_Unified"]
    numeric = ["Age_Median"]

    observed_summary = pd.DataFrame([survival_summary(analysis, "Observed T3/T4 radiation-treated cohort")])
    covariates = ["Age_Median", "Sex", "T_Unified", "N_Unified", "Stage_Group", "Histology_Unified"]
    matched_df = propensity_score_match(analysis, covariates)
    matched_summary = pd.DataFrame([survival_summary(matched_df, "Propensity-score matched cohort")])

    _, model_metrics = train_outcome_model(analysis, features + ["Chemotherapy_Binary"], categorical, numeric + ["Chemotherapy_Binary"])
    benefit_df = estimate_individual_benefit(analysis, features, categorical, numeric)
    subgroup_results = subgroup_table(benefit_df)
    trial_patients, trial_results, bootstrap_df = simulate_trial(benefit_df, args.trial_n, args.bootstrap, args.seed)

    cohort_characteristics = (
        analysis.groupby("Treatment_Arm_Observed")
        .agg(
            N=("Age_Median", "size"),
            Age_Mean=("Age_Median", "mean"),
            Overall_Survival_Median_Months=("Overall_Survival_Months", "median"),
            Cancer_Mortality_Percent=("Cancer_Death_Event_Capped", lambda s: s.mean() * 100),
        )
        .reset_index()
        .rename(columns={"Treatment_Arm_Observed": "Observed_Treatment"})
    )

    observed_summary.to_csv(tables_dir / "observed_t3t4_chemo_benefit.csv", index=False)
    matched_summary.to_csv(tables_dir / "psm_t3t4_chemo_benefit.csv", index=False)
    cohort_characteristics.to_csv(tables_dir / "cohort_characteristics.csv", index=False)
    model_metrics.to_csv(tables_dir / "predictive_model_performance.csv", index=False)
    subgroup_results.to_csv(tables_dir / "predicted_benefit_by_subgroup.csv", index=False)
    trial_results.to_csv(tables_dir / "in_silico_trial_results.csv", index=False)
    trial_patients.to_csv(tables_dir / "simulated_trial_patients.csv", index=False)

    web_payload = {
        "clinical_question": "Does adding chemotherapy to radiotherapy improve survival in T3/T4 disease?",
        "trial_n": args.trial_n,
        "observed_summary": observed_summary.to_dict(orient="records"),
        "matched_summary": matched_summary.to_dict(orient="records"),
        "model_metrics": model_metrics.to_dict(orient="records"),
        "trial_results": trial_results.to_dict(orient="records"),
        "top_predicted_benefit_subgroups": subgroup_results.head(20).to_dict(orient="records"),
    }
    (tables_dir / "publication_results.json").write_text(json.dumps(web_payload, indent=2), encoding="utf-8")

    write_summary(
        reports_dir / "publication_summary.txt",
        analysis,
        observed_summary,
        matched_summary,
        model_metrics,
        trial_results,
    )
    write_focused_in_silico_outputs(
        focused_trial_dir,
        analysis,
        trial_patients,
        trial_results,
        bootstrap_df,
        model_metrics,
    )

    print("Analysis completed successfully.")
    print(f"Tables: {tables_dir}")
    print(f"Summary: {reports_dir / 'publication_summary.txt'}")
    print(f"Focused in-silico trial output: {focused_trial_dir}")


if __name__ == "__main__":
    main()
