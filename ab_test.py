"""
A/B Testing, Simplified (plain Python version, no Streamlit)

A step-by-step A/B test analysis on a real e-commerce dataset:
old landing page (A) vs new landing page (B).

Run:
    pip install pandas numpy scipy
    python ab_test.py                      # uses ab_data.csv in the same folder
    python ab_test.py path/to/your.csv     # your own data, same 5 columns

Optional: install matplotlib to also save a daily conversion chart (daily_conversion.png).

Part of the LinkedIn series "Data & Analytics, Simplified" by Victor Taiwo.
Do the analysis. Explain it like a human.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05          # significance level
POWER = 0.80          # for the sample size calculation
MDE = 0.002           # smallest lift worth detecting, absolute (0.002 = 0.2 percentage points)
REQUIRED = {"user_id", "timestamp", "group", "landing_page", "converted"}


def heading(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = REQUIRED - set(df.columns)
    if missing:
        sys.exit(f"Your CSV is missing these columns: {', '.join(sorted(missing))}")
    return df


def clean_data(df):
    """Step 1: drop rows where group and page disagree. Step 2: keep each user's earliest visit."""
    mismatch = (df["group"] == "treatment") != (df["landing_page"] == "new_page")
    matched = df[~mismatch]
    dup_users = int(matched["user_id"].duplicated().sum())
    clean = matched.sort_values("timestamp").drop_duplicates("user_id", keep="first")
    return clean, int(mismatch.sum()), dup_users


def two_proportion_test(x_b, n_b, x_a, n_a, alpha=ALPHA):
    """Pooled two-proportion z-test (two-sided) and an unpooled confidence interval for B - A."""
    p_a, p_b = x_a / n_a, x_b / n_b
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    return {"p_a": p_a, "p_b": p_b, "diff": diff, "z": z, "p": p_value,
            "lo": diff - z_crit * se_diff, "hi": diff + z_crit * se_diff}


def sample_size_per_group(p0, mde, alpha=ALPHA, power=POWER):
    """Users needed in EACH group to detect an absolute lift of `mde` (two-sided test)."""
    p1 = p0 + mde
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p0 + p1) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return int(np.ceil(num / mde ** 2))


def save_daily_chart(daily, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed, skipping the chart)")
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(daily.index, daily["control"] * 100, color="#2a78d6", lw=2, marker="o", ms=3, label="A · old page")
    ax.plot(daily.index, daily["treatment"] * 100, color="#eb6834", lw=2, marker="o", ms=3, label="B · new page")
    ax.set_ylabel("Daily conversion rate (%)")
    ax.set_title("Daily conversion rate: A vs B")
    ax.grid(axis="y", color="#e1e0d9")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved chart: {path}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "ab_data.csv"
    raw = load_data(path)

    heading("A/B TESTING, SIMPLIFIED")
    print("A/B testing = comparing two versions to see which one performs better.")
    print("A = old landing page (control), B = new landing page (treatment).")

    # 1. What data do we have?
    heading("1. What data do we actually have?")
    print(f"Rows:           {len(raw):,}")
    print(f"Unique users:   {raw['user_id'].nunique():,}")
    print(f"Days of data:   {raw['timestamp'].dt.date.nunique()} "
          f"({raw['timestamp'].min():%d %b %Y} to {raw['timestamp'].max():%d %b %Y})")
    print(f"Missing values: {int(raw.isna().sum().sum())}")
    print("\nFirst rows:")
    print(raw.head().to_string(index=False))

    # 2. Can we trust it?
    heading("2. Can we trust it?")
    print("Every A user should see the old page and every B user the new page.\n")
    print(pd.crosstab(raw["group"], raw["landing_page"]).to_string())
    clean, n_mismatch, n_dup = clean_data(raw)
    print(f"\nRows on the wrong page for their group: {n_mismatch:,} ({n_mismatch / len(raw):.1%}) -> removed")
    print(f"Users counted more than once:          {n_dup} -> kept the earliest visit")
    print(f"Clean dataset:                         {len(clean):,} unique users")

    # 3. Was the split fair?
    heading("3. Were users split fairly?")
    counts = clean["group"].value_counts().reindex(["control", "treatment"])
    srm = stats.chisquare(counts.values)
    print(f"A (old page): {counts['control']:,}  ({counts['control'] / counts.sum():.1%})")
    print(f"B (new page): {counts['treatment']:,}  ({counts['treatment'] / counts.sum():.1%})")
    print(f"Chi-square check for a lopsided split (SRM): p = {srm.pvalue:.2f}")
    print("-> The split is fair." if srm.pvalue >= 0.01 else
          "-> WARNING: the split is lopsided. Investigate assignment before trusting results.")

    # 4. Conversion rate
    heading("4. Conversion rate")
    g = clean.groupby("group")["converted"].agg(users="count", conversions="sum")
    x_a, n_a = int(g.loc["control", "conversions"]), int(g.loc["control", "users"])
    x_b, n_b = int(g.loc["treatment", "conversions"]), int(g.loc["treatment", "users"])
    res = two_proportion_test(x_b, n_b, x_a, n_a)
    print(f"A (old page): {x_a:,} of {n_a:,} converted = {res['p_a']:.2%}")
    print(f"B (new page): {x_b:,} of {n_b:,} converted = {res['p_b']:.2%}")
    print(f"Difference (B - A): {res['diff'] * 100:+.2f} percentage points "
          f"({res['diff'] / res['p_a']:+.1%} relative)")

    # 5. Real gap or chance?
    heading("5. Is the gap real, or just chance?")
    print("Two-proportion z-test: if the pages were truly identical, how likely is a gap this big?")
    print(f"z = {res['z']:.2f}, p-value = {res['p']:.3f}")
    print(f"{1 - ALPHA:.0%} confidence interval for B - A: "
          f"{res['lo'] * 100:+.2f} to {res['hi'] * 100:+.2f} percentage points")
    significant = res["p"] < ALPHA
    if significant:
        print(f"-> p < {ALPHA}: evidence that B performs {'better' if res['diff'] > 0 else 'worse'} than A.")
    else:
        print(f"-> p >= {ALPHA} and the interval crosses zero: no evidence of a real difference.")

    # 6. Over time
    heading("6. Look at it over time")
    daily = (clean.assign(date=clean["timestamp"].dt.date)
             .groupby(["date", "group"])["converted"].mean().unstack())
    b_wins = int((daily["treatment"] > daily["control"]).sum())
    print(f"B beat A on {b_wins} of {len(daily)} days.")
    print((daily * 100).round(2).rename(columns={"control": "A %", "treatment": "B %"}).to_string())
    save_daily_chart(daily, Path(__file__).parent / "daily_conversion.png")

    # 7. Sample size
    heading("7. How many users would the next test need?")
    n_needed = sample_size_per_group(res["p_a"], MDE)
    print(f"To detect a lift of {MDE * 100:.1f} percentage points from a {res['p_a']:.2%} baseline")
    print(f"(alpha = {ALPHA}, power = {POWER:.0%}): {n_needed:,} users per group.")
    print(f"This test had about {min(n_a, n_b):,} users per group.")

    # Conclusion
    heading("CONCLUSION")
    if significant:
        print(f"The difference is statistically significant at alpha = {ALPHA}.")
        print("Before acting, check the lift is big enough to matter for the business.")
    else:
        print(f"The new page converted at {res['p_b']:.2%} vs {res['p_a']:.2%} for the old page.")
        print(f"With p = {res['p']:.2f}, there is no evidence the new page is better.")
        print("Keep the old page, or rethink the redesign. 'No difference' is still a result.")


if __name__ == "__main__":
    main()
