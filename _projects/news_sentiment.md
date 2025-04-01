---
name: Financial News Sentiment Analysis Project
tools: [Python, NLP, FinBERT, AlphaVantage, GDELTDoc, Data Visualization]
image: /assets/images/financial-sentiment-analysis.png
description: Sentiment analysis of 6,584 financial and economic news articles collected from January 1 to March 20, 2025 using keyword-based search
---

# Financial News Sentiment Analysis Project

## Project Overview

This project analyzes the sentiment of 6,584 news articles collected from January 1 to March 20, 2025, using specific keyword groups related to economic, political, and financial topics. This represents the first phase of a larger research initiative that will eventually combine news sentiment with stock price data to explore correlations between media sentiment and market movements.

## Data Collection Methodology

Articles were collected using AlphaVantage API and GDELTDoc, focusing on 10 keyword groups:

- Trump_Trade_Policy
- Russia_Ukraine_Conflict  
- Monetary_Policy
- Market_Indicators
- Economic_Indicators
- Financial_Markets
- Tech_Sector
- Global_Trade
- Energy_Oil
- Stock_Market

Each keyword group contained specific search terms to ensure comprehensive coverage of relevant news.

## Sentiment Analysis Approach

The collected articles were processed using FinBERT, a BERT-based model specifically fine-tuned for financial sentiment analysis. This model provides more accurate sentiment classification for financial texts compared to general-purpose sentiment analyzers, distinguishing between positive, negative, and neutral sentiments with greater precision in financial contexts.

## Key Findings

(This section will detail the key findings from your sentiment analysis)

## Future Work

The next phase of this project will combine the sentiment analysis results with stock price data to investigate potential correlations between news sentiment and market movements. This comprehensive analysis will help understand how media sentiment might influence or reflect market behavior across different sectors and timeframes.

## Tech Stack

- Python: Primary programming language
- AlphaVantage API: Financial news data source
- GDELTDoc: Global news monitoring
- FinBERT: Financial sentiment analysis model
- Pandas: Data manipulation and analysis
- Matplotlib/Seaborn: Data visualization
- Jupyter Notebook: Analysis documentation

[View Complete Analysis](#) <!-- Link to be added later -->
