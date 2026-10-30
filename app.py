"""
A/B Testing, Simplified
A step-by-step A/B test analysis on a real e-commerce dataset.
Part of the "Data & Analytics, Simplified" series by Victor Taiwo.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(page_title="A/B Testing, Simplified", page_icon="🧪", layout="wide")

COLOR_A = "#2a78d6"  # old page (control)
COLOR_B = "#eb6834"  # new page (treatment)
DATA_PATH = Path(__file__).parent / "ab_data.csv"


# ---------------------------------------------------------------- helpers
@st.cache_data
def load_data(path_or_buffer):
    df = pd.read_csv(path_or_buffer, parse_dates=["timestamp"])
    return df


def clean_data(df):
    """Remove rows where the group and the page shown disagree, then de-duplicate users."""
    mismatch = (df["group"] == "treatment") != (df["landing_page"] == "new_page")
    matched = df[~mismatch]
    dup_users = int(matched["user_id"].duplicated().sum())
    clean = matched.sort_values("timestamp").drop_duplicates("user_id", keep="first")
    return clean, int(mismatch.sum()), dup_users


def two_proportion_test(x_b, n_b, x_a, n_a, alpha):
    """Two-proportion z-test (pooled) plus an unpooled confidence interval for B - A."""
    p_a, p_b = x_a / n_a, x_b / n_b
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pool
    p_two = 2 * (1 - stats.norm.cdf(abs(z)))
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    return dict(p_a=p_a, p_b=p_b, diff=diff, z=z, p=p_two,
                lo=diff - z_crit * se_diff, hi=diff + z_crit * se_diff)


def sample_size_per_group(p0, mde_abs, alpha=0.05, power=0.8):
    """Users needed in EACH group to detect an absolute lift of mde_abs (two-sided test)."""
    p1 = p0 + mde_abs
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p0 + p1) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return int(np.ceil(num / mde_abs ** 2))


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
alpha = st.sidebar.select_slider("Significance level (α)", options=[0.01, 0.05, 0.10], value=0.05,
                                 help="How much risk of a false alarm you accept. 0.05 is the usual choice.")
uploaded = st.sidebar.file_uploader("Use your own CSV (optional)", type="csv",
                                    help="Needs columns: user_id, timestamp, group, landing_page, converted")
st.sidebar.markdown("---")
st.sidebar.markdown("**Data & Analytics, Simplified**  \nDo the analysis. Explain it like a human.")

# ---------------------------------------------------------------- data
try:
    raw = load_data(uploaded if uploaded is not None else DATA_PATH)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the data: {e}")
    st.stop()

required = {"user_id", "timestamp", "group", "landing_page", "converted"}
if not required.issubset(raw.columns):
    st.error(f"Your CSV needs these columns: {', '.join(sorted(required))}")
    st.stop()

clean, n_mismatch, n_dup = clean_data(raw)

# ---------------------------------------------------------------- header
st.title("🧪 A/B Testing, Simplified")
st.markdown(
    "**A/B testing = comparing two versions to see which one performs better.** "
    "Here we follow a real e-commerce test, step by step: an **old landing page (A)** "
    "against a **new landing page (B)**. No jumping straight to the answer."
)

# ---------------------------------------------------------------- step 1
st.header("1 · What data do we actually have?")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(raw):,}")
c2.metric("Unique users", f"{raw['user_id'].nunique():,}")
days = raw["timestamp"].dt.date.nunique()
c3.metric("Days of data", days, help=f"{raw['timestamp'].min():%d %b %Y} to {raw['timestamp'].max():%d %b %Y}")
c4.metric("Missing values", int(raw.isna().sum().sum()))
st.caption("One row per visit: who the user is, when they came, which group they were assigned, "
           "which page they actually saw, and whether they converted (1) or not (0).")
st.dataframe(raw.head(8), width="stretch", hide_index=True)

# ---------------------------------------------------------------- step 2
st.header("2 · Can we trust it?")
st.markdown("Every user in group A (control) should have seen the **old** page, and every user "
            "in group B (treatment) the **new** page. Let's check.")
ct = pd.crosstab(raw["group"], raw["landing_page"]).rename_axis(index="Group", columns="Page shown")
st.dataframe(ct, width="stretch")
st.warning(
    f"**{n_mismatch:,} rows ({n_mismatch / len(raw):.1%})** show a user on the wrong page for their group. "
    "We can't tell which experience they really got, so we remove them. "
    f"We also found **{n_dup} user(s) counted more than once**, and keep only their first visit."
)
st.success(f"Clean dataset: **{len(clean):,} unique users**.")

# ---------------------------------------------------------------- step 3
st.header("3 · Were users split fairly?")
counts = clean["group"].value_counts().reindex(["control", "treatment"])
chi = stats.chisquare(counts.values)
c1, c2, c3 = st.columns(3)
c1.metric("A · old page", f"{counts['control']:,}")
c1.caption(f"{counts['control'] / counts.sum():.1%} of users")
c2.metric("B · new page", f"{counts['treatment']:,}")
c2.caption(f"{counts['treatment'] / counts.sum():.1%} of users")
c3.metric("Split check (p-value)", f"{chi.pvalue:.2f}")
if chi.pvalue < 0.01:
    st.error("The split is lopsided (Sample Ratio Mismatch). Something went wrong in assignment. "
             "Investigate before trusting any result.")
else:
    st.info("A chi-square test checks whether the 50/50 split is believable. "
            "A high p-value means **the split is fair**, so we can move on.")

# ---------------------------------------------------------------- step 4
st.header("4 · Conversion rate")
g = clean.groupby("group")["converted"].agg(users="count", conversions="sum")
x_a, n_a = int(g.loc["control", "conversions"]), int(g.loc["control", "users"])
x_b, n_b = int(g.loc["treatment", "conversions"]), int(g.loc["treatment", "users"])
res = two_proportion_test(x_b, n_b, x_a, n_a, alpha)
c1, c2, c3 = st.columns(3)
c1.metric("A · old page", f"{res['p_a']:.2%}")
c1.caption(f"{x_a:,} of {n_a:,} converted")
c2.metric("B · new page", f"{res['p_b']:.2%}")
c2.caption(f"{x_b:,} of {n_b:,} converted")
c3.metric("Difference (B − A)", f"{res['diff'] * 100:+.2f} pp")
c3.caption(f"{res['diff'] / res['p_a']:+.1%} relative")
st.caption("Conversion rate = people who converted ÷ people in the group. "
           "pp = percentage points.")

# ---------------------------------------------------------------- step 5
st.header("5 · Is the gap real, or just chance?")
st.markdown(
    f"A **two-proportion z-test** asks: *if the two pages were truly identical, how likely is a gap this big?* "
    f"That likelihood is the **p-value**. The **{1 - alpha:.0%} confidence interval** is a reasonable range "
    "for the true difference."
)
c1, c2 = st.columns(2)
c1.metric("p-value", f"{res['p']:.3f}")
c2.metric(f"{1 - alpha:.0%} confidence interval (B − A)", f"{res['lo'] * 100:+.2f} to {res['hi'] * 100:+.2f} pp")

ci_df = pd.DataFrame({"label": ["B − A"], "diff": [res["diff"] * 100],
                      "lo": [res["lo"] * 100], "hi": [res["hi"] * 100]})
span = max(abs(res["lo"]), abs(res["hi"])) * 100 * 1.6
base = alt.Chart(ci_df).encode(y=alt.Y("label:N", title=None, axis=None))
ci_chart = (
    alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#0b0b0b", strokeWidth=2).encode(x="x:Q")
    + base.mark_rule(color=COLOR_B, strokeWidth=4).encode(
        x=alt.X("lo:Q", title="Difference in conversion rate, B − A (percentage points)",
                scale=alt.Scale(domain=[-span, span])), x2="hi:Q",
        tooltip=[alt.Tooltip("lo:Q", title="Lower", format="+.2f"), alt.Tooltip("hi:Q", title="Upper", format="+.2f")])
    + base.mark_point(filled=True, size=220, color=COLOR_B).encode(
        x="diff:Q", tooltip=[alt.Tooltip("diff:Q", title="B − A (pp)", format="+.2f")])
).properties(height=110)
st.altair_chart(ci_chart, width="stretch")

significant = res["p"] < alpha
if significant and res["diff"] > 0:
    st.success(f"p < {alpha}: evidence that **B performs better** than A.")
elif significant and res["diff"] < 0:
    st.error(f"p < {alpha}: evidence that **B performs worse** than A.")
else:
    st.info(f"p ≥ {alpha} and the interval crosses zero: **no evidence of a real difference.** "
            "The gap we see is consistent with random noise.")

# ---------------------------------------------------------------- step 6
st.header("6 · Look at it over time")
daily = (clean.assign(date=clean["timestamp"].dt.date)
         .groupby(["date", "group"])["converted"].mean().reset_index())
daily["Page"] = daily["group"].map({"control": "A · old page", "treatment": "B · new page"})
wide = daily.pivot(index="date", columns="group", values="converted")
b_wins = int((wide["treatment"] > wide["control"]).sum())
line = (
    alt.Chart(daily).mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=40))
    .encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("converted:Q", title="Daily conversion rate", axis=alt.Axis(format=".0%"),
                scale=alt.Scale(zero=False)),
        color=alt.Color("Page:N", scale=alt.Scale(domain=["A · old page", "B · new page"],
                                                    range=[COLOR_A, COLOR_B]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("date:T", title="Date"), "Page:N",
                 alt.Tooltip("converted:Q", title="Conversion", format=".2%")],
    ).properties(height=320)
)
st.altair_chart(line, width="stretch")
st.caption(f"B beat A on {b_wins} of {len(wide)} days. When the lines keep crossing, that is noise, not a winner.")

# ---------------------------------------------------------------- step 7
st.header("7 · Plan your next test: how many users do you need?")
st.markdown("A test can only find a difference it is big enough to detect. Decide the smallest lift "
            "you care about **before** you test.")
c1, c2 = st.columns(2)
baseline = c1.number_input("Current conversion rate (%)", 0.5, 90.0, round(res["p_a"] * 100, 2), 0.1) / 100
mde = c2.number_input("Smallest lift worth detecting (percentage points)", 0.05, 20.0, 0.5, 0.05) / 100
n_needed = sample_size_per_group(baseline, mde, alpha=alpha)
st.metric("Users needed per group (80% power)", f"{n_needed:,}")
st.caption(f"This test had about {min(n_a, n_b):,} users per group.")

# ---------------------------------------------------------------- conclusion
st.header("✅ Conclusion")
if not significant:
    st.markdown(
        f"The new page converted at **{res['p_b']:.2%}** vs **{res['p_a']:.2%}** for the old page. "
        f"With p = {res['p']:.2f}, there is **no evidence the new page is better**. "
        "Keep the old page, or rethink the redesign. \"No difference\" is still a result: "
        "it stops a company from rolling out a change that wouldn't pay off."
    )
else:
    st.markdown(f"The difference is statistically significant at α = {alpha}. "
                "Before acting, check that the lift is big enough to matter for the business.")

st.markdown("---")
st.caption("Data: public e-commerce A/B test dataset (Udacity Data Analyst Nanodegree), January 2017. "
           "Built by Victor Taiwo for the LinkedIn series *Data & Analytics, Simplified*.")
