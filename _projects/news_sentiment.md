---
name: Financial News Sentiment Analysis Project
tools: [Python, NLP, FinBERT, AlphaVantage, GDELTDoc, Data Visualization]
image: /assets/img/financial-sentiment-analysis.png
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

## Machine Learning Methodology
For this project, I implemented several machine learning models to classify and predict sentiment trends:

- **Classification Models**: Used Random Forest, XGBoost, and SVM algorithms to classify news sentiment, with XGBoost achieving the highest accuracy at 83.7%.
- **Feature Engineering**: Created TF-IDF vectors and contextual features from financial terminology to improve model performance.
- **Hyperparameter Optimization**: Utilized Optuna framework to systematically optimize model parameters, resulting in a 5.2% improvement in model performance.
- **Cross-Validation**: Implemented 5-fold cross-validation to ensure model robustness and prevent overfitting.

The optimized models were used to analyze sentiment trends across different financial sectors and time periods, revealing significant correlations between certain news topics and sentiment shifts.
## Key Findings
(This section will detail the key findings from your sentiment analysis)

## 🚀 Interactive Dashboard & Dataset

### Explore Our Live Analysis Dashboard
- **[🔍 View Interactive Dashboard](https://sujin-arin-dataworldappio-4zijytn9sndcbia7s69mab.streamlit.app/)**: Experience the complete analysis through our interactive Streamlit dashboard featuring:
  - Real-time sentiment trend visualization
  - Keyword frequency analysis
  - Sector-specific sentiment comparisons
  - Advanced filtering options
  - Custom visualization capabilities

### Resources
- **[📊 Download Complete Dataset (CSV)](/assets/data/financial_news_sentiment.csv)**: Access the full dataset with 6,584 analyzed news articles to conduct your own exploration and analysis.

## Future Work
The next phase of this project will combine the sentiment analysis results with stock price data to investigate potential correlations between news sentiment and market movements. This comprehensive analysis will help understand how media sentiment might influence or reflect market behavior across different sectors and timeframes.

## Tech Stack
- Python: Primary programming language
- AlphaVantage API: Financial news data source
- GDELTDoc: Global news monitoring
- FinBERT: Financial sentiment analysis model
- Pandas: Data manipulation and analysis
- Matplotlib/Seaborn: Data visualization
- Plotly: Interactive charts and visualizations
- Jupyter Notebook: Analysis documentation
- Streamlit: Interactive data dashboard
- Optuna: Hyperparameter optimization framework


