---
name: Financial News Sentiment Analysis Project
tools: [Python, NLP, FinBERT, AlphaVantage, GDELTDoc, Data Visualization]
image: /assets/img/financial-sentiment-analysis.png
description: Sentiment analysis of 9,235 financial and economic news articles collected from 01.01.23 to 12.31.24 using keyword-based search
---

# Financial News Sentiment Analysis Project

## Project Overview
This project analyzes the sentiment of 9,235 news articles collected from January 2023 to December 2024, using specific keyword groups related to economic, political, and financial topics. This represents the first phase of a larger research initiative that will eventually combine news sentiment with stock price data to explore correlations between media sentiment and market movements.

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

Each keyword group contained specific search terms to ensure comprehensive coverage of relevant news. The collection process involved daily API calls with rate limiting to avoid throttling, followed by deduplication and content validation steps.

### Data Preprocessing
- Cleaned HTML and special characters from article text
- Removed duplicate articles based on title and content similarity
- Filtered out articles shorter than 100 words
- Applied language detection to ensure English-only content
- Extracted publication dates, sources, and categories for segmentation

## Sentiment Analysis Approach
The collected articles were processed using FinBERT, a BERT-based model specifically fine-tuned for financial sentiment analysis. This model provides more accurate sentiment classification for financial texts compared to general-purpose sentiment analyzers, distinguishing between positive, negative, and neutral sentiments with greater precision in financial contexts.

### Sentiment Distribution
- **Positive articles**: 32.7% (3,020 articles)
- **Neutral articles**: 45.8% (4,230 articles)
- **Negative articles**: 21.5% (1,985 articles)

## Machine Learning Methodology
For this project, I explored machine learning models to identify patterns between news sentiment and other features:

- **Models Explored**: Implemented RandomForest and GradientBoosting classifiers to categorize market movements
- **Feature Engineering**: Created temporal features and integrated sentiment indicators with economic data
- **Hyperparameter Exploration**: Used Optuna framework to test different parameter combinations
- **Cross-Validation**: Implemented validation techniques to evaluate generalizability

The models helped identify potential relationships between sentiment patterns and market indicators, though further refinement is needed to improve predictive capabilities.

## Key Findings

### Topic-Based Sentiment Analysis
- **Tech Sector**: Demonstrated the most positive sentiment overall, particularly around AI advancements and semiconductor industry news
- **Russia-Ukraine Conflict**: Showed predominantly negative sentiment, with brief positive shifts during ceasefire discussions
- **Monetary Policy**: Exhibited cyclical sentiment patterns aligned with Federal Reserve announcements

### Temporal Patterns
- Identified a lag between major news events and subsequent sentiment shifts
- Detected weekday vs. weekend sentiment variation, with weekend news showing more extreme sentiment (both positive and negative)
- Observed increased sentiment volatility during earnings seasons

### Source Analysis
- Financial news from specialized sources showed more consistent sentiment classification
- General news outlets demonstrated higher sentiment volatility on the same topics

## 🚀 Interactive Dashboard & Dataset

### Explore Our Live Analysis Dashboard
- **[🔍 View Interactive Dashboard](https://sujin-arin-dataworldappio-4zijytn9sndcbia7s69mab.streamlit.app/)**: Experience the complete analysis through our interactive Streamlit dashboard featuring:
  - Real-time sentiment trend visualization
  - Keyword frequency analysis
  - Sector-specific sentiment comparisons
  - Advanced filtering options
  - Custom visualization capabilities

### Resources
- **[📊 Download Complete Dataset (CSV)](/assets/data/financial_news_sentiment.csv)**: Access the full dataset with 9,235 analyzed news articles to conduct your own exploration and analysis.

## Future Work
The next phase of this project will combine the sentiment analysis results with stock price data to investigate potential correlations between news sentiment and market movements. This comprehensive analysis will help understand how media sentiment might influence or reflect market behavior across different sectors and timeframes.

### Planned Enhancements
- Refinement of machine learning models to improve prediction accuracy
- Integration with additional market data sources
- Implementation of more sophisticated time-series analysis techniques
- Development of sector-specific sentiment indicators
- Exploration of relationships between news events and market movements
- Creation of a sentiment index for major market sectors

## Tech Stack
- **Python**: Primary programming language
- **AlphaVantage API**: Financial news data source
- **GDELTDoc**: Global news monitoring
- **FinBERT**: Financial sentiment analysis model
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Plotly**: Interactive charts and visualizations
- **Jupyter Notebook**: Analysis documentation
- **Streamlit**: Interactive data dashboard
- **Optuna**: Hyperparameter exploration framework
