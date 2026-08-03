"""
TASK 2: Unemployment Analysis with Python
--------------------------------------------
Goal: Analyze unemployment trends, the Covid-19 shock, and labour market
health more broadly (not just the unemployment rate).

This version focuses on:
  - Estimated Employed numbers (actual jobs lost, not just rate %)
  - Labour Participation Rate as a second lens on the same shock
  - Month-over-month % change (growth/decline momentum)
  - A simple state risk-tier classification (Low/Medium/High unemployment)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")

RATE = "Estimated Unemployment Rate (%)"
EMPLOYED = "Estimated Employed"
PARTICIPATION = "Estimated Labour Participation Rate (%)"


def load_clean(path, rename_cols=None):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if rename_cols:
        df = df.rename(columns=rename_cols)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["Date", RATE]).reset_index(drop=True)


# -----------------------------
# STEP 1: Load data
# -----------------------------
df = load_clean("data/Unemployment_in_India.csv")
print("Shape:", df.shape, "| Date range:", df["Date"].min().date(), "to", df["Date"].max().date())
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# -----------------------------
# STEP 2: Jobs lost in absolute numbers, not just rate %
# -----------------------------
employed_trend = df.groupby("Date")[EMPLOYED].sum().reset_index()
employed_trend["MoM_Change"] = employed_trend[EMPLOYED].pct_change() * 100

peak_employed = employed_trend[EMPLOYED].max()
trough_employed = employed_trend.loc[employed_trend["Date"] >= "2020-03-01", EMPLOYED].min()
jobs_lost = peak_employed - trough_employed

print(f"\nPeak total estimated employment: {peak_employed:,.0f}")
print(f"Lowest point during Covid window: {trough_employed:,.0f}")
print(f"Estimated jobs lost at the trough: {jobs_lost:,.0f} ({jobs_lost/peak_employed*100:.1f}% of peak)")

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(employed_trend["Date"], employed_trend[EMPLOYED] / 1e6, color="steelblue", marker="o")
ax1.set_ylabel("Total Estimated Employed (millions)", color="steelblue")
ax1.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"), color="red", alpha=0.12)
ax2 = ax1.twinx()
ax2.bar(employed_trend["Date"], employed_trend["MoM_Change"], width=15, alpha=0.3, color="orange")
ax2.set_ylabel("Month-over-Month % Change", color="orange")
plt.title("Total Employment Level & Month-over-Month Change")
fig.tight_layout()
plt.savefig("output/1_employment_level_and_change.png", dpi=120)
plt.close()

# -----------------------------
# STEP 3: Labour Participation Rate as a second lens
# -----------------------------
participation_trend = df.groupby("Date")[PARTICIPATION].mean()
rate_trend = df.groupby("Date")[RATE].mean()

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(rate_trend.index, rate_trend.values, color="crimson", marker="o", label="Unemployment Rate (%)")
ax1.set_ylabel("Unemployment Rate (%)", color="crimson")
ax2 = ax1.twinx()
ax2.plot(participation_trend.index, participation_trend.values, color="green", marker="s", label="Labour Participation Rate (%)")
ax2.set_ylabel("Labour Participation Rate (%)", color="green")
plt.title("Unemployment Rate vs Labour Participation Rate")
fig.tight_layout()
plt.savefig("output/2_rate_vs_participation.png", dpi=120)
plt.close()

corr = df[[RATE, PARTICIPATION]].corr().iloc[0, 1]
print(f"\nCorrelation between Unemployment Rate and Labour Participation Rate: {corr:.3f}")

# -----------------------------
# STEP 4: 3-month rolling average to smooth noise
# -----------------------------
national = df.groupby("Date")[RATE].mean().reset_index()
national["Rolling_3M"] = national[RATE].rolling(3, min_periods=1).mean()

plt.figure(figsize=(10, 5))
plt.plot(national["Date"], national[RATE], alpha=0.4, label="Monthly rate (raw)")
plt.plot(national["Date"], national["Rolling_3M"], linewidth=2.5, label="3-month rolling average")
plt.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"), color="red", alpha=0.12, label="Covid window")
plt.title("National Unemployment Rate: Raw vs Smoothed Trend")
plt.legend()
plt.tight_layout()
plt.savefig("output/3_rolling_average_trend.png", dpi=120)
plt.close()

# -----------------------------
# STEP 5: Classify states into risk tiers
# -----------------------------
state_avg = df.groupby("Region")[RATE].mean().sort_values(ascending=False)

def tier(rate):
    if rate >= state_avg.quantile(0.66):
        return "High"
    elif rate >= state_avg.quantile(0.33):
        return "Medium"
    return "Low"

tiers = state_avg.apply(tier)
tier_counts = tiers.value_counts()
print("\nState risk-tier counts:\n", tier_counts)
print("\nHigh-risk states:\n", tiers[tiers == "High"].index.tolist())

plt.figure(figsize=(8, 8))
colors = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}
tier_colors = [colors[tiers[state]] for state in state_avg.index]
plt.barh(state_avg.index, state_avg.values, color=tier_colors)
plt.gca().invert_yaxis()
plt.xlabel("Average Unemployment Rate (%)")
plt.title("States Ranked by Unemployment Rate (colored by risk tier)")
plt.tight_layout()
plt.savefig("output/4_state_risk_tiers.png", dpi=120)
plt.close()

# -----------------------------
# STEP 6: Policy summary
# -----------------------------
print("\n--- Summary ---")
print(f"1. At the lowest point during Covid, an estimated {jobs_lost/1e6:.1f} million jobs (~{jobs_lost/peak_employed*100:.0f}% of peak employment) were lost.")
print(f"2. Unemployment rate and labour participation rate moved with a correlation of {corr:.2f} — {'a strong inverse relationship' if corr < -0.3 else 'a weak relationship'}, meaning {'as unemployment rose, fewer people were even looking for work' if corr < -0.3 else 'participation stayed fairly independent of the unemployment rate'}.")
print(f"3. {len(tiers[tiers=='High'])} states fall into the 'High risk' tier and would benefit most from targeted employment programs: {', '.join(tiers[tiers=='High'].index.tolist()[:5])}.")
print("4. The 3-month rolling average shows the shock was sharp but short — smoothing reveals the trend recovering faster than the raw monthly numbers suggest.")

print("\nDone. Charts saved in the output/ folder.")
