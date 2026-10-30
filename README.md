# A/B Testing, Simplified

**A/B testing = comparing two versions to see which one performs better.**

This is an interactive, step-by-step analysis of a real e-commerce A/B test: an **old landing page (A)** against a **new landing page (B)**, using 294,478 user visits over 23 days. It walks through the analysis the way an analyst should, with no jumping straight to the answer:

1. What data do we actually have?
2. Can we trust it? (users on the wrong page, duplicate users)
3. Were users split fairly? (Sample Ratio Mismatch check)
4. What is the conversion rate of each page?
5. Is the gap real, or just chance? (two-proportion z-test, confidence interval)
6. How does it look over time?
7. How many users would the next test need? (sample size calculator)

Part of the LinkedIn series **Data & Analytics, Simplified** by [Victor Taiwo](https://www.linkedin.com/in/victor-taiwo-076094167/).
*Do the analysis. Explain it like a human.*

## Why this matters

A/B tests drive real decisions: which page to launch, which price to charge, which feature to keep. Many bad A/B conclusions come from skipping the boring checks, not from bad statistics. This project shows that:

- **Data quality comes first.** 3,893 rows (1.3%) had users on the wrong page for their group. Analysing them would mix up the two experiences.
- **A visible gap is not automatically a real one.** The new page converted at 11.88% against 12.04% for the old page, but with p = 0.19 the difference is consistent with random noise.
- **"No difference" is still a result.** It stops a team from rolling out a redesign that wouldn't pay off.
- **Sample size should be planned before the test.** Detecting very small lifts needs far more users than most people expect.

## Results on the included dataset

| Step | Finding |
|---|---|
| Rows / unique users | 294,478 rows, 290,584 users after cleaning |
| Data quality | 3,893 mismatched rows removed, 1 duplicate user removed |
| Split check | A: 145,274, B: 145,310 (50/50, chi-square p = 0.95) |
| Conversion rate | A (old page): 12.04%, B (new page): 11.88% |
| Difference (B − A) | −0.16 percentage points, 95% CI −0.39 to +0.08 |
| Significance | Two-proportion z-test, p = 0.19 (not significant) |
| Sample size check | Detecting a +0.2 pp lift (80% power, α = 0.05) needs about 418,500 users per group; this test had about 145,300 |
| Conclusion | No evidence the new page converts better |

## Use cases

- **Learning:** a clear, end-to-end example for students and young data professionals learning A/B testing.
- **Teaching and mentoring:** a ready-made walkthrough for workshops or study groups.
- **Reusable template:** upload your own CSV with the same columns to run the same checks and tests on your experiment.
- **Test planning:** use the sample size calculator to estimate how many users a future test needs.

## Dataset

`ab_data.csv` is a public e-commerce A/B test dataset widely used in the Udacity Data Analyst Nanodegree ("Analyze A/B Test Results") and on Kaggle. It covers 2–24 January 2017.

| Column | Meaning |
|---|---|
| `user_id` | Unique user identifier |
| `timestamp` | When the user visited |
| `group` | Assigned group: `control` (A) or `treatment` (B) |
| `landing_page` | Page actually shown: `old_page` or `new_page` |
| `converted` | 1 if the user converted, 0 if not |

To use your own data, upload a CSV with these five columns from the app's sidebar.

## Environment

- Python 3.10 or 3.11 recommended
- Libraries: Streamlit, pandas, NumPy, SciPy, Altair (see `requirements.txt`)
- No API keys, database or internet connection needed: the dataset ships with the repository

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository, for example `ab-testing-simplified`.
2. Upload `app.py`, `requirements.txt`, `README.md` and `ab_data.csv` to the repository root.
3. Visit https://share.streamlit.io/ and choose **Create app**.
4. Select the repository, branch (`main`) and entrypoint `app.py`, then deploy.

Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Method notes

- **Cleaning, in this order:** (1) rows where `group` and `landing_page` disagree are removed (3,893 rows); (2) for any user still appearing more than once, only their earliest visit by `timestamp` is kept (1 user, whose two visits had the same group and outcome). Result: 290,584 unique users. All later numbers use this cleaned dataset.
- **Split check:** chi-square goodness-of-fit test against an equal split.
- **Significance test:** pooled two-proportion z-test, two-sided, α = 0.05 by default (adjustable in the sidebar).
- **Confidence interval:** unpooled normal-approximation interval for the difference B − A.
- **Sample size:** standard two-proportion formula (two-sided, α = 0.05, 80% power), using the old page's conversion rate (12.04%) as the baseline. It is a planning calculation for future tests, not a post-hoc power estimate.
- **Assumptions:** users were randomly assigned, each user is counted once, and users' outcomes are independent of each other.

## Limitations

- The dataset has no information on traffic source, device or country, so segment-level effects can't be explored.
- The test assumes users were randomly assigned and that one user's behaviour doesn't affect another's.
- The analysis uses a single fixed-horizon test; it does not cover sequential testing or Bayesian methods.
- Results describe this historical dataset only and should not be generalised to other sites.
