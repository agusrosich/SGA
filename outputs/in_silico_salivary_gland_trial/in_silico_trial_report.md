# Salivary Gland Cancer In-Silico Trial Results

## Clinical Question
Does adding chemotherapy to radiotherapy improve overall survival in patients with advanced T3/T4 salivary gland cancer?

## In-Silico Trial Design
- Design: RTOG 1008-like randomized in-silico trial.
- Population: T3/T4 salivary gland cancer patients treated with radiation alone or concurrent chemoradiation in the source cohort.
- Eligible retrospective cohort: 954 patients.
- Simulated sample size: 252 patients.
- Randomization: 1:1 chemoradiation vs radiation alone.
- Endpoint: predicted overall survival in months, administratively capped at 120 months.
- Uncertainty: 1,000 bootstrap simulations.

## Predictive Model
- Model: Gradient boosting mortality classifier.
- Endpoint used for model performance: 10-year capped cancer-specific mortality.
- Apparent AUC: 0.806.
- Cross-validated AUC: 0.677 +/- 0.017.

## Primary Simulated Trial Result
- Chemoradiation arm: 126 patients.
- Radiation-alone arm: 126 patients.
- Median predicted OS with chemoradiation: 41.8 months.
- Median predicted OS with radiation alone: 41.5 months.
- Median OS difference: +0.3 months.
- Log-rank p-value: 0.1187.
- Bootstrap mean OS difference: -0.3 months.
- Bootstrap 95% CI: -3.9 to +3.3 months.
- Probability that OS difference is greater than 0: 0.436.

## Interpretation
In-silico trial does not predict a statistically robust overall survival benefit from adding chemotherapy.

This output is restricted to the in-silico salivary gland cancer trial. It should be interpreted as a retrospective predictive simulation and hypothesis-generating evidence, not as prospective causal proof.
