---
name: Fear & Greed Market Analysis Dashboard
tools: [Python, Plotly, Data Analysis]
image: /assets/img/fear_greed_preview.png
description: An interactive dashboard analyzing the relationship between market sentiment (Fear & Greed Index) and stock performance across different sectors.
category: finance
featured: true
---

# Fear & Greed Market Analysis Dashboard

This project analyzes the relationship between the Fear & Greed Index and stock market performance. Based on data from January 2023 to March 2025, it visualizes which stocks perform well during fear and greed market phases.

<div class="embed-responsive">
  <iframe src="/assets/fear_greed_dashboard.html" height="800px" width="100%" frameborder="0" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);"></iframe>
</div>

{% include elements/button.html link="/assets/fear_greed_dashboard.html" text="Explore Interactive Dashboard" %}

## Key Features

- Identification of stocks that perform well during fear periods (Fear & Greed Index ≤ 25)
- Analysis of stocks that decline after greed periods (Fear & Greed Index ≥ 75)
- Volume analysis by stock groups
- Interactive filtering by sector groups

## Stocks and Sectors Analyzed

The dashboard analyzes the following stocks across different sectors:

| Ticker | Company/ETF | Sector Group |
|--------|-------------|--------------|
| TSLA | Tesla | Technology |
| NVDA | NVIDIA | Technology |
| JPM | JPMorgan Chase | Financials |
| SPY | S&P 500 ETF | Financials |
| XOM | Exxon Mobil | Commodities |
| X | US Steel | Commodities |
| ALB | Albemarle | Commodities |
| WMT | Walmart | Consumer Staples |
| CAT | Caterpillar | Industrials |
| GLD | Gold ETF | Safe-Haven |

The analysis groups stocks into sectors to identify how different parts of the market respond to changes in market sentiment. This helps reveal patterns such as:

- Which sectors perform best during periods of market fear
- Which stocks show resilience during sentiment shifts
- How trading volume across sectors changes with market sentiment

## Key Findings

- Technology stocks show approximately 15% better returns during extreme fear periods
- Safe-haven assets like gold (GLD) outperform during periods of market greed
- Consumer staples demonstrate the most stable performance regardless of market sentiment
- Trading volume increases significantly across all sectors during sentiment extremes


