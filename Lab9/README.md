# Recovering Experimental Truths via Propensity Score Matching

## Objective
Demonstrate that rigorous econometric methods — specifically Propensity Score Matching — can recover an unbiased causal treatment effect from observational data that is otherwise corrupted by severe selection bias.

---

## Methodology

- **Modeled Selection Bias:** Characterized the selection mechanism driving treatment assignment using the observational subset of the Lalonde dataset, where control units were drawn from general population surveys (CPS/PSID) rather than a randomized control group — a setup known to produce extreme confounding.

- **Estimated Propensity Scores:** Trained a logistic regression model on pre-treatment covariates — including age, education, race, marital status, degree attainment, and prior earnings (1974, 1975) — to estimate each unit's conditional probability of receiving treatment. This collapses a high-dimensional confounder space into a single scalar score.

- **Applied Nearest Neighbor Matching:** For each treated unit, identified the closest control unit in propensity score space using a 1-to-1 Euclidean nearest neighbor algorithm. The resulting matched dataset approximates the covariate balance of a randomized experiment, isolating the average treatment effect on the treated (ATT).

---

## Key Findings

| Estimate | Method | Result |
|---|---|---|
| Naive Difference in Means | Raw observational comparison | **-$635** |
| PSM-Adjusted Estimate | Nearest Neighbor Matching | **+$1,800** |
| Benchmark (RCT) | Lalonde (1986) experimental ground truth | **~$1,794** |

The naive comparison — an uncontrolled difference in mean 1978 earnings between treated and control groups — produced a **negative** estimate of -$635, falsely implying that job training *reduced* earnings. This is a textbook artifact of selection bias: individuals who sought training were systematically disadvantaged relative to the general-population control group, making the raw comparison meaningless as a causal estimate.

After matching on propensity scores to construct a balanced counterfactual control group, the estimated treatment effect converged to approximately **+$1,800** — nearly identical to the experimental benchmark established by Lalonde (1986) using a randomized controlled trial. This result confirms that PSM successfully neutralized the confounding and recovered the true causal effect of job training on earnings.

---

## Tools & Stack
`Python` · `Pandas` · `Scikit-Learn` · `Statsmodels` · `Seaborn`

---

*Reference: Lalonde, R.J. (1986). "Evaluating the Econometric Evaluations of Training Programs with Experimental Data." American Economic Review, 76(4), 604–620.*
