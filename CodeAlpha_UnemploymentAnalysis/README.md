# CodeAlpha_UnemploymentAnalysis

Unemployment Analysis with Python — CodeAlpha Data Science Internship (Task 2)

## Overview
Goes beyond the headline unemployment rate to look at actual jobs lost, labour participation, smoothed trends, and a state-level risk classification.

## Dataset
`Unemployment_in_India.csv` — May 2019 to Jun 2020, with Region, Area (Rural/Urban), Estimated Unemployment Rate, Estimated Employed, and Estimated Labour Participation Rate.

## Approach
1. Cleaned the data (whitespace/BOM handling, date parsing)
2. Tracked **absolute jobs lost** (Estimated Employed) alongside month-over-month % change
3. Compared Unemployment Rate against Labour Participation Rate as two different lenses on the same shock
4. Applied a 3-month rolling average to separate signal from noise
5. Classified all states into **Low / Medium / High** unemployment risk tiers based on their average rate

## Results
- At the lowest point during Covid, an estimated **137.4 million jobs (~34% of peak employment)** were lost
- Unemployment rate and labour participation rate showed almost **no correlation (0.003)** — participation didn't simply track the rate, suggesting people weren't uniformly dropping out of the workforce as unemployment rose
- **10 states** fall into the High-risk tier, including Tripura, Haryana, Jharkhand, Bihar, and Himachal Pradesh
- The 3-month rolling average shows the shock was sharp but short-lived — smoothing reveals a faster underlying recovery than the raw monthly numbers suggest

## Files
```
unemployment_analysis.py                # main script
data/Unemployment_in_India.csv          # dataset
output/                                 # generated charts (employment level, rate vs participation, rolling trend, state risk tiers)
```

## How to run
```bash
pip install pandas numpy matplotlib seaborn
python unemployment_analysis.py
```

## Tech stack
Python, Pandas, NumPy, Matplotlib, Seaborn

---
*Part of the CodeAlpha Data Science Internship*
