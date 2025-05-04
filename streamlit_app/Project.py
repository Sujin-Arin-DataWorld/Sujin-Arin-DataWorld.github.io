import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           roc_curve, auc)
import optuna
import warnings
import matplotlib.dates as mdates
from scipy import stats
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_correlations(data, category, feature_cols, num_indicators):
    """상관관계 계산 결과를 캐싱하는 함수"""
    correlations = data[feature_cols].corrwith(data[category]).abs().sort_values(ascending=False)
    return correlations.head(num_indicators)

# 페이지 설정
st.set_page_config(page_title="StockFlow AI: Reading Stocks Markets with Sentiment and Econmic Indicators", layout="wide")

# 시각화 설정
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['axes.grid'] = True

# 메인 타이틀
st.title("StockFlow AI: Aktienmärkte verstehen mit Stimmungs- und Wirtschaftsindikatoren")
st.markdown("""
# Vorhersage von Aktienkursbewegungen mit Machine Learning 🤖

---
            
### 💡 Was diese App macht:

- 📈 **Sammelt tägliche Aktienkursdaten**, kategorisiert nach Sektoren (z. B. Technologie, Energie, Gold, Verteidigung)
- 📰 **Sammelt Finanznachrichten** mithilfe **kategoriespezifischer Schlüsselwörter**
- 🧠 **Wendet das FinBERT-Modell an**, um Wahrscheinlichkeiten für positive, neutrale und negative Stimmungen aus Nachrichtenartikeln zu extrahieren
- 🔥 **Integriert externe Finanzindikatoren** wie den **Fear & Greed Index**, **10-jährige Staatsanleihenrenditen**, **Dollar-Index** und **makroökonomische Kennzahlen**
- ⚙️ **Normalisiert alle Indikatoren**, um einen direkten Vergleich zwischen unterschiedlichen Datentypen zu ermöglichen
- 🧪 **Erstellt Machine-Learning-Modelle**, um **Aufwärts- oder Abwärtsbewegungen der Aktienkurse** in jeder Kategorie vorherzusagen
- 🚨 **Analysiert extreme Ereignisse**, wie starke Kurseinbrüche, ungewöhnliche Volatilität und Stimmungseinbrüche auf Basis der Machine-Learning-Ergebnisse
- 🧠 **Generiert Marktanalysen**, indem Nachrichtentrends, wirtschaftliche Bedingungen und Modellprognosen kombiniert werden

---

Entdecke, wie Marktstimmungen und Wirtschaftsindikatoren die Zukunft der Aktienmärkte beeinflussen!

👉 **Zuerst**: Führe die **Machine-Learning-Vorhersage** aus, um Kursbewegungen auf Basis von Stimmungen und wirtschaftlichen Indikatoren zu analysieren.

👉 **Anschließend**: Erkunde **Extreme Events & Anomalien**, um tiefere Einblicke in Markterschütterungen und Ausreißer zu erhalten.
            
---
### 📚 Datenquellen:

- 📈 **Aktienmarktdaten:** yfinance  
- 📚 **Nachrichtensammlung:** GDELT-Bibliothek  
- 😨 **Fear & Greed Index:** Alternative.me Fear & Greed Index API  
- 📊 **Wirtschaftsindikatoren:** FRED (Federal Reserve Economic Data)          
""")

st.markdown("---")

# 컬럼 나누기
col1, col2 = st.columns(2)

# 왼쪽: Ticker 소개
with col1:
    st.subheader("🏷️ Ticker Overview by Category")
    
    ticker_data = {
        "Category": ["Tech & AI", "Energy", "Gold & Mining", "Defense & Aerospace"],
        "Tickers": [
            "Apple, Miscrosoft, NVIDIA, Palantir",
            "ExxonMobil, Chevron, BP, Shell",
            "Gold ETF, Gold Miners ETF, Agnico Eagle, Newmont",
            "Lockheed Martin, TX Corporation, Boeing, Northrop Grumman"
        ],
        "Description": [
            "Technology and AI Leaders",
            "Major Energy and Oil Companies",
            "Gold ETFs and Mining Stocks",
            "Defense and Aerospace Giants"
        ]
    }
    ticker_df = pd.DataFrame(ticker_data)
    st.dataframe(ticker_df, use_container_width=True)

    st.markdown("""
        ### 🛠️ Feature Engineering Überblick
        - **Zeitreihen-Features:** Lag 1/3/5 Tage, Rolling Mean/Std 5/10/20 Tage, Momentum, Volatilität
        - **Technische Indikatoren:** MA5, MA20, MA50, RSI, MACD
        - **Korrelationen:** 90/5/20/50 Tage zwischen Aktienkursen und Wirtschaftsindikatoren
        """)
    st.success("""
    - **Starke Veränderungen makroökonomischer Indikatoren:** Plötzliche Anstiege oder Rückgänge von BIP, Verbraucherpreisindex (CPI) usw.
    - **Black-Swan-Ereignisse:** Unvorhersehbare Marktverwerfungen (starker Einbruch oder starker Anstieg)
    - **Korrelationseinbruch:** Zusammenbruch traditioneller Korrelationen zwischen Sektoren
    - **Technische Muster:** Erkennung wichtiger Signale wie Golden Cross und Death Cross
    """)

# 오른쪽: 경제 지표 & 감성 지표 소개
with col2:
    st.subheader("📊 Economic and Sentiment Indicators")
    
    indicator_data = {
    "Type": ["Macroeconomic"] * 11 + ["Sentiment"] * 4, 
    "Indicator": [
        "GDP_norm", "CPI_norm", "Industrial_Production_norm", "Real_Interest_Rate_norm",
        "Consumer_Sentiment_norm", "WTI_Oil_norm",
        "10Y_Treasury_Yield", "Natural_Gas", "Dollar_Index", "Government_Spending",
        "Fed_Funds_Rate_norm",  # 추가
        "Positive/Neutral/Negative Probabilities (FinBERT)", 
        "Sentiment Mean", 
        "Sentiment Variance",
        "Fear & Greed Index"
    ],
    "Description": [
        "Gross Domestic Product Growth (Normalized)",
        "Consumer Price Index Inflation (Normalized)",
        "Industrial Production Rate (Normalized)",
        "Real Interest Rate Movements (Normalized)",
        "Consumer Sentiment Index (Normalized)",
        "Oil Price Movements (WTI Crude Oil, Normalized)",
        "10-Year Treasury Yield (Normalized)",
        "Natural Gas Prices (Normalized)",
        "US Dollar Strength Index (Normalized)",
        "Government Spending Index (Normalized)",
        "**Federal Funds Rate (Normalized)**",   # 설명도 추가
        "Probabilities from Financial News Sentiment Analysis",
        "Average Sentiment Score across news articles",
        "Variance of Sentiment Scores",
        "Market Greed or Fear Sentiment Index"
    ]
}
    indicator_df = pd.DataFrame(indicator_data)
    st.dataframe(indicator_df, use_container_width=True)
    
    st.success("""
    ➔ Über die reine Analyse historischer Daten hinaus haben wir eine **fortschrittliche Feature-Engineering-Strategie** entwickelt,  
    die auch Marktsentiment und extreme Ereignisse berücksichtigt.
    """)
   

st.markdown("---")

st.markdown("""
✅ Alle Indikatoren sind **normalisiert**, um die Vergleichbarkeit zu gewährleisten.  
✅ Die Sentiment-Analyse wird mithilfe von **FinBERT** auf Finanznachrichten durchgeführt.  
✅ Echte Wirtschaftsindikatoren wie **Staatsanleihenrenditen**, **Erdgaspreise** und **Dollar-Index** sind für bessere Vorhersagen integriert.
""")

df = load_data()

# 유틸리티 함수들
# 경제 및 감성 지표 컬럼 목록 (전역 변수로 정의)
economic_columns = [
    'Consumer_Sentiment',
    'CPI',
    'Dollar_Index',
    'Fed_Funds_Rate',
    'GDP',
    'Gov_Spending',
    'Industrial_Production',
    'Natural_Gas',
    'Real_Interest_Rate',
    'WTI_Oil',
    '10Y_Treasury_Yield'
]

sentiment_columns = [
    'fear_greed_value',
    'sentiment_score_mean',
    'positive_prob_mean',
    'negative_prob_mean',
    'neutral_prob_mean',
    'sentiment_group_Zscore'
]
#'daily_sentiment_mean_first'

def create_output_directory(dir_name='ML_improved_results'):
    """결과 저장을 위한 디렉토리를 생성합니다."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # 카테고리별 하위 디렉토리 생성
    categories_dir = os.path.join(dir_name, 'categories')
    if not os.path.exists(categories_dir):
        os.makedirs(categories_dir)
    return dir_name
def preprocess_data(df):
    """데이터 전처리 및 특성 공학 - Z-score 정규화 포함"""
    # 필수 컬럼 확인
    df= df.copy()
    categories =df['category'].unique()
    ticker = df['ticker'].unique()
    required_columns = ['ticker', 'date', 'close']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"필수 컬럼 '{col}'이 데이터에 없습니다. 올바른 데이터 파일을 업로드하세요.")
            return df

    category_tickers = {}
    for category in categories:
        category_tickers[category] = df[df['category'] == category]['ticker'].unique()
        print(f"{category} Categories's Ticker: {category_tickers[category]}")
    # 카테고리 컬럼이 없으면 추가
    if 'category' not in df.columns:
        # 티커에 따라 카테고리 매핑
        category_map = {}
        for category, tickers in category_tickers.items():
            for ticker in tickers:
                category_map[ticker] = category
        
        # 카테고리 컬럼 추가
        df['category'] = df['ticker'].map(category_map)
        df['category'].fillna('Other', inplace=True)
    
    # 일일 수익률 계산
    df['daily_return'] = df.groupby('ticker')['close'].pct_change() * 100

    df['sentiment_score_std'].fillna(df['sentiment_score_std'].median(), inplace=True)
    df['daily_sentiment_std_first'].fillna(df['daily_sentiment_std_first'].median(), inplace=True)


    # 경제지표와 감성지표 컬럼 정의
    economic_columns = ['Consumer_Sentiment', 'CPI', 'Dollar_Index', 'Fed_Funds_Rate',
                    'GDP', 'Gov_Spending', 'Industrial_Production', 'Natural_Gas',
                    'Real_Interest_Rate', 'WTI_Oil', '10Y_Treasury_Yield']

    sentiment_columns = ['fear_greed_value', 'sentiment_score_mean', 'positive_prob_mean',
                        'negative_prob_mean', 'neutral_prob_mean', ]#'daily_sentiment_mean_first'

    # 경제지표 정규화 - Z-score 방식
    for col in economic_columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[f'{col}_norm'] = (df[col] - mean_val) / std_val

    # 경제지표 정규화 - Min-Max 방식 (필요시)
    for col in economic_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[f'{col}_minmax'] = (df[col] - min_val) / (max_val - min_val)

    # 감성지표 정규화 - Z-score 방식
    for col in sentiment_columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[f'{col}_norm'] = (df[col] - mean_val) / std_val

    # 감성지표 정규화 - Min-Max 방식 (필요시)
    for col in sentiment_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[f'{col}_minmax'] = (df[col] - min_val) / (max_val - min_val)

    # 카테고리별 감성 Z-score 정규화
    df['category_means'] = df.groupby('category')['sentiment_score_mean'].transform('mean')
    df['category_stds'] = df.groupby('category')['sentiment_score_mean'].transform('std')
    df['sentiment_group_Zscore'] = (df['sentiment_score_mean'] - df['category_means']) / df['category_stds']

    # 정규화된 컬럼명 리스트 생성
    norm_economic_columns = [f'{col}_norm' for col in economic_columns]
    norm_sentiment_columns = [f'{col}_norm' for col in sentiment_columns] + ['sentiment_group_Zscore']

    # 카테고리별 종가 평균 계산 (기존 코드)
    daily_avg_prices = df.groupby(['date', 'category'])['close'].mean().unstack()

    # 정규화된 지표들로 일별 지표 테이블 생성
    daily_indicators = df.groupby('date')[norm_economic_columns + norm_sentiment_columns].first()   
        # 나머지 코드는 그대로 유지...
        
    return df, daily_avg_prices, daily_indicators

def create_lag_features(df, column, lags=[1, 3, 5, 10, 20]):
    """시계열 데이터에 대한 지연 특성을 생성합니다."""
    df = df.copy()
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df

def create_rolling_features(df, column, windows=[5, 10, 20, 50]):
    """롤링 통계 특성을 생성합니다."""
    df = df.copy()
    for window in windows:
        # 롤링 평균
        df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
        # 롤링 표준편차
        df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
        # 롤링 변화율 (현재값/평균값)
        df[f'{column}_rolling_ratio_{window}'] = df[column] / df[f'{column}_rolling_mean_{window}']
    return df

def create_momentum_features(df, column, periods=[5, 10, 20, 50]):
    """모멘텀(변화율) 특성을 생성합니다."""
   
    for period in periods:
        # 단순 변화율
        df[f'{column}_change_{period}d'] = df[column].pct_change(periods=period)
        # 가속도(변화율의 변화율)
        df[f'{column}_acceleration_{period}d'] = df[f'{column}_change_{period}d'].pct_change(periods=period)
    return df

def create_volatility_features(df, column, windows=[5, 10, 20, 50]):
    """변동성 지표를 생성합니다."""
   
    for window in windows:
        # 롤링 변동성 (표준편차 기반)
        df[f'{column}_volatility_{window}d'] = df[column].pct_change().rolling(window=window).std()
        # 상대 변동성 (평균 대비)
        rolling_mean = df[column].rolling(window=window).mean()
        rolling_std = df[column].rolling(window=window).std()
        df[f'{column}_rel_volatility_{window}d'] = rolling_std / rolling_mean
    return df

def create_technical_indicators(df, column):
    """기술적 지표를 생성합니다."""
   
    # 이동평균선 (Moving Averages)
    df[f'{column}_MA5'] = df[column].rolling(window=5).mean()
    df[f'{column}_MA20'] = df[column].rolling(window=20).mean()
    df[f'{column}_MA50'] = df[column].rolling(window=50).mean()
    # 이동평균 교차 신호
    df[f'{column}_MA_cross_5_20'] = np.where(df[f'{column}_MA5'] > df[f'{column}_MA20'], 1, -1)
    df[f'{column}_MA_cross_20_50'] = np.where(df[f'{column}_MA20'] > df[f'{column}_MA50'], 1, -1)
    # RSI (Relative Strength Index) 계산 - 14일 기준
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df[f'{column}_RSI_14'] = 100 - (100 / (1 + rs))
    # MACD (Moving Average Convergence Divergence)
    ema12 = df[column].ewm(span=12, adjust=False).mean()
    ema26 = df[column].ewm(span=26, adjust=False).mean()
    df[f'{column}_MACD'] = ema12 - ema26
    df[f'{column}_MACD_signal'] = df[f'{column}_MACD'].ewm(span=9, adjust=False).mean()
    df[f'{column}_MACD_hist'] = df[f'{column}_MACD'] - df[f'{column}_MACD_signal']

    return df

def create_interaction_features(df, columns):
    """변수 간 상호작용 특성을 생성합니다."""

    # 주요 특성 간 상호작용
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            # 곱셈 상호작용
            df[f'{col1}_mul_{col2}'] = df[col1] * df[col2]
            # 비율 상호작용
            if not (df[col2] == 0).any():  # 0으로 나누기 방지
                df[f'{col1}_div_{col2}'] = df[col1] / df[col2]
            # 합 상호작용
            df[f'{col1}_add_{col2}'] = df[col1] + df[col2]
    return df

def create_correlation_features(df, category_tickers, windows=[90, 5, 20, 50]):
    """
    롤링 상관관계 특성 생성 (카테고리-지표 간 및 카테고리-카테고리 간)
    """
    # 데이터 검증 및 디버깅 정보
    #st.write("상관관계 특성 생성을 위한 데이터 구조 확인:")
    #st.write(f"DataFrame 컬럼: {df.columns.tolist()[:10]}... 등 {len(df.columns)}개")
    
    # 필요한 컬럼 확인
    required_cols = ['date', 'category', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"필요한 컬럼이 누락되었습니다: {missing_cols}")
        return df, None, None
    
    # 날짜 형식 확인
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        st.warning("'date' 컬럼이 datetime 형식이 아닙니다. 변환을 시도합니다.")
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            st.error(f"날짜 변환 오류: {str(e)}")
            return df, None, None
    
    # 카테고리별 일별 평균 가격 계산
    try:
        daily_avg_prices = df.groupby(['date', 'category'])['close'].mean().unstack()
        #st.write(f"카테고리별 일별 평균 가격 계산 완료: {daily_avg_prices.shape}")
    except Exception as e:
        st.error(f"일별 평균 가격 계산 오류: {str(e)}")
        return df, None, None
    
    # 정규화된 경제 및 감성 지표 컬럼 식별
    norm_economic_columns = [f'{col}_norm' for col in economic_columns if f'{col}_norm' in df.columns]
    norm_sentiment_columns = [f'{col}_norm' for col in sentiment_columns if f'{col}_norm' in df.columns]
    
    # 경고: 정규화된 지표가 없는 경우
    if not norm_economic_columns and not norm_sentiment_columns:
        st.warning("정규화된 지표가 발견되지 않았습니다. '_norm' 형식의 컬럼이 필요합니다.")
        # 존재하는 컬럼 중 가능한 지표 컬럼 표시
        possible_indicators = [col for col in df.columns if any(econ in col for econ in economic_columns) or 
                              any(sent in col for sent in sentiment_columns)]
        #if possible_indicators:
        #     st.write(f"가능한 지표 컬럼: {possible_indicators}")
    
    # 모든 정규화된 지표 컬럼 (+ sentiment_group_Zscore)
    all_norm_columns = norm_economic_columns + norm_sentiment_columns
    if 'sentiment_group_Zscore' in df.columns:
        all_norm_columns.append('sentiment_group_Zscore')
    
    #st.write(f"사용할 정규화된 지표 컬럼: {len(all_norm_columns)}개")
    
    # 일별 지표 데이터 (첫 번째 행 기준)
    if all_norm_columns:
        try:
            daily_indicators = df.groupby('date')[all_norm_columns].first()
            #st.write(f"일별 지표 데이터 계산 완료: {daily_indicators.shape}")
        except Exception as e:
            st.error(f"일별 지표 계산 오류: {str(e)}")
            return df, daily_avg_prices, None
    else:
        st.error("정규화된 지표가 없어 일별 지표 데이터를 생성할 수 없습니다.")
        return df, daily_avg_prices, None
    
    # 상관관계 계산을 위한 빈 데이터프레임 생성
    corr_df = pd.DataFrame(index=daily_avg_prices.index)
    
    # 1. 중요 카테고리-지표 조합에 대한 90일 상관관계 계산
    important_combinations = [
        ('Gold', 'GDP_norm'),
        ('Gold', 'Gov_Spending_norm'),
        ('Gold', 'Real_Interest_Rate_norm'),
        ('Defense', 'Dollar_Index_norm'),
        ('Defense', 'Industrial_Production_norm'),
        ('Energy', 'Consumer_Sentiment_norm'),
        ('Energy', 'Dollar_Index_norm'),
        ('Energy', 'Gov_Spending_norm'),
        ('Tech_AI', 'Dollar_Index_norm')
    ]
    
    # 처리 가능한 중요 조합 필터링
    available_combinations = []
    for category, indicator in important_combinations:
        if category in daily_avg_prices.columns and indicator in daily_indicators.columns:
            available_combinations.append((category, indicator))
    
   # st.write(f"Important combinations that can be processed: {len(available_combinations)} / {len(important_combinations)}")
    
    # 중요 조합에 대해 90일 롤링 상관계수 계산
    for category, indicator in available_combinations:
        price_series = daily_avg_prices[category]
        indicator_series = daily_indicators[indicator]
        
        # 원본과 동일한 이름 사용
        corr_name = f"{category}_{indicator}_corr_mean90"
        corr_df[corr_name] = price_series.rolling(window=90).corr(indicator_series)
    
    # 2. 카테고리-카테고리 및 추가 윈도우 상관관계 계산
    categories = list(daily_avg_prices.columns)
    reference_columns = []
    
    # 참조 컬럼 설정 - 카테고리 + 상위 경제지표
    if categories:
        reference_columns.extend(categories)
    
    if norm_economic_columns:
        reference_columns.extend(norm_economic_columns[:min(9, len(norm_economic_columns))])
    
    #st.write(f"Reference Columns: {reference_columns}")
    
    # 각 카테고리와 참조 컬럼에 대해 추가 윈도우 상관관계 계산
    correlations_created = 0
    
    for category in categories:
        price_series = daily_avg_prices[category]
        
        for ref_col in reference_columns:
            if ref_col == category:  # 자기 자신과의 상관관계는 계산하지 않음
                continue
                
            # 이미 중요 조합으로 90일 상관관계가 계산되었는지 확인
            is_important_90d = False
            for imp_cat, imp_ind in available_combinations:
                if category == imp_cat and ref_col == imp_ind:
                    is_important_90d = True
                    break
            
            # 참조 시리즈 가져오기
            ref_series = None
            if ref_col in categories:
                ref_series = daily_avg_prices[ref_col]
            elif ref_col in daily_indicators.columns:
                ref_series = daily_indicators[ref_col]
            
            if ref_series is not None:
                # 원래 윈도우 목록에서 필요한 것만 계산
                for window in windows:
                    # 90일 윈도우이고 이미 중요 조합으로 계산된 경우 건너뛰기
                    if window == 90 and is_important_90d:
                        continue
                    
                    # 원본과 동일한 이름 규칙 사용
                    if window == 90 and (category, ref_col) in available_combinations:
                        # 중요 조합은 이미 위에서 처리됨
                        continue
                    else:
                        corr_name = f"corr_{category}_{ref_col}_{window}d"
                        corr_df[corr_name] = price_series.rolling(window=window).corr(ref_series)
                        correlations_created += 1
    
   # st.write(f"Created correlation attribution: {correlations_created}개")
    
    # 상관관계 특성이 없으면 경고
    if corr_df.empty:
        st.warning("생성된 상관관계 특성이 없습니다.")
        return df, daily_avg_prices, daily_indicators
    
    # 날짜별로 상관관계 값을 원본 데이터프레임에 추가
    merged_count = 0
    for corr_col in corr_df.columns:
        # 데이터프레임을 재설정하여 날짜를 컬럼으로 변환
        temp_df = corr_df[corr_col].reset_index()
        temp_df.columns = ['date', corr_col]
        
        # 원본 데이터프레임과 병합
        df = pd.merge(df, temp_df, on='date', how='left')
        merged_count += 1
    
    #st.write(f"Merged correlation attributes: {merged_count}")
    
    # 결측치 처리
    for col in corr_df.columns:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill')  # 앞의 값으로 채우기
            df[col] = df[col].fillna(0)  # 남은 결측치는 0으로
    
    return df, daily_avg_prices, daily_indicators

def get_indicator_columns():
    """분석에 사용할 지표 컬럼명 목록을 반환합니다."""
    # 정규화된 경제 지표 변수 목록
    norm_economic_columns = [f'{col}_norm' for col in economic_columns]
    
    # 정규화된 감성 지표 변수 목록
    norm_sentiment_columns = [f'{col}_norm' for col in sentiment_columns]
    
    # 정규화된 추가 특성 목록
    additional_features = [
        'Gold_GDP_norm_corr_mean90',
        'Gold_Gov_Spending_norm_corr_mean90',
        'Gold_Real_Interest_Rate_norm_corr_mean90',
        'Defense_Dollar_Index_norm_corr_mean90',
        'Defense_Industrial_Production_norm_corr_mean90',
        'Energy_Consumer_Sentiment_norm_corr_mean90',
        'Energy_Dollar_Index_norm_corr_mean90',
        'Energy_Gov_Spending_norm_corr_mean90',
        'Tech_AI_Dollar_Index_norm_corr_mean90'
    ]
    
    # 모든 정규화된 지표 변수 목록
    all_norm_indicators = norm_economic_columns + norm_sentiment_columns + additional_features
    
    return norm_economic_columns, norm_sentiment_columns, additional_features, all_norm_indicators
def prepare_enhanced_model_data(daily_avg_prices, daily_indicators, categories, important_indicators):
    """향상된 모델링 데이터를 준비하는 함수 (무한대, 극단값 안전 처리 통합 버전)"""
    import numpy as np
    import pandas as pd
    import streamlit as st

    model_data = {}
    if 'visualizations' not in st.session_state:
        st.session_state['visualizations'] = {}

    # 입력 데이터 검증
    if daily_avg_prices is None or daily_indicators is None:
        st.error("유효한 가격 데이터 또는 지표 데이터가 없습니다.")
        return model_data

    # 일치하는 날짜 확인
    common_dates = daily_avg_prices.index.intersection(daily_indicators.index)
    if len(common_dates) == 0:
        st.error("가격 데이터와 지표 데이터 간에 일치하는 날짜가 없습니다.")
        return model_data

    #st.write(f"모델 데이터 준비: 일치하는 날짜 {len(common_dates)}개")

    # 진행 상황 표시
    #progress_bar = st.progress(0)
    #status_text = st.empty()

    available_categories = [cat for cat in categories if cat in daily_avg_prices.columns]
    if len(available_categories) < len(categories):
        st.warning(f"일부 카테고리를 찾을 수 없습니다. 사용 가능한 카테고리: {available_categories}")

    data_loss_diagnostics = {}

    for i, category in enumerate(available_categories):
        #status_text.text(f"{category} Category's Model Data is on the way...")

        try:
            price_series = daily_avg_prices[category]

            # 안전한 로그 수익률 계산 (0은 NaN으로 처리)
            price_series = price_series.replace(0, np.nan)
            log_returns = np.log(price_series / price_series.shift(1))

            available_indicators = [col for col in important_indicators if col in daily_indicators.columns]
            if len(available_indicators) == 0:
                st.warning(f"{category}에 대한 사용 가능한 지표가 없습니다.")
                continue

            # 기본 모델 데이터 생성
            model_df = pd.concat([
                log_returns.rename(category),
                daily_indicators[available_indicators]
            ], axis=1)

            # --- 무한대와 극단값 처리 추가 시작 ---
            model_df = model_df.replace([np.inf, -np.inf], np.nan)  # 무한대 제거
            FLOAT32_MAX = np.finfo(np.float32).max
            model_df = model_df.clip(lower=-FLOAT32_MAX, upper=FLOAT32_MAX)  # 극단값 클리핑
            # --- 무한대와 극단값 처리 추가 끝 ---

            # NaN 처리
            essential_cols = [category]
            if model_df[category].isna().any():
                model_df[category] = model_df[category].fillna(method='ffill').fillna(method='bfill')

            # 필수 lag 변수들 처리 (필요 시)
            lag_cols = [col for col in model_df.columns if col.startswith(f"{category}_lag") or col.startswith(f"{category}_rolling")]
            for col in lag_cols:
                model_df[col] = model_df[col].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')

            # 나머지 수치형 변수들 처리
            numeric_cols = model_df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                if model_df[col].isna().any():
                    median_value = model_df[col].median()
                    if pd.isna(median_value):
                        model_df[col] = model_df[col].fillna(0)
                    else:
                        model_df[col] = model_df[col].fillna(median_value)

            # 최종적으로 남은 NaN은 0으로
            model_df = model_df.fillna(0)

            # 최종 데이터 검증
            if len(model_df) < 30:
                st.warning(f"{category}: 특성 생성 후 데이터가 부족합니다 ({len(model_df)} rows). 최소 30개 필요.")
                continue

            #st.write(f"{category} 강화된 데이터 준비 완료: {len(model_df)}행, {len(model_df.columns)}열")
            model_data[category] = model_df

        except Exception as e:
            st.error(f"{category} 모델 데이터 준비 중 오류: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

        # 진행 상황 업데이트
        #progress_bar.progress((i + 1) / len(available_categories))

    #status_text.text(f"모델 데이터 준비 완료: {len(model_data)}개 카테고리")
    #progress_bar.empty()

    if not model_data:
        st.error("모든 카테고리에 대한 모델 데이터 생성에 실패했습니다.")

    return model_data



def select_best_features(X, y, feature_names, n_features=30):
    """랜덤 포레스트를 사용하여 최적의 특성을 선택합니다."""
    try:
        # 특성 중요도를 계산하기 위한 랜덤 포레스트 학습
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # 특성 중요도에 따라 특성 정렬
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # 상위 n개 특성 선택
        selected_features = [feature_names[i] for i in indices[:n_features]]
        
        return selected_features
    except Exception as e:
        st.error(f"특성 선택 중 오류: {str(e)}")
        return feature_names[:min(n_features, len(feature_names))]  # 오류 시 처음 n개 특성 반환


def find_optimal_threshold(y_true, y_proba):
    """
    ROC 곡선과 다양한 메트릭을 고려하여 최적의 임계값을 찾습니다.
    다양한 방법을 시도하고 가장 안정적인 임계값을 반환합니다.
    """
    try:
        # NaN 또는 무한값 체크 및 처리
        if np.isnan(y_proba).any() or np.isinf(y_proba).any():
            y_proba = np.nan_to_num(y_proba, nan=0.5, posinf=1.0, neginf=0.0)
        
        # 클래스 비율 확인
        class_ratio = np.mean(y_true)
        
        # 각 방법으로 계산한 임계값을 저장할 리스트
        thresholds_list = []
        
        # 1. ROC 곡선 기반 방법 (유클리디안 거리)
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_proba)
        
        # 완벽한 분류 지점 (0,1)에서 가장 가까운 지점 찾기 (유클리디안 거리)
        distances = np.sqrt((1-tpr)**2 + fpr**2)
        optimal_idx_roc = np.argmin(distances)
        
        # 거리가 0.8보다 크면 ROC 기반 임계값이 불안정할 수 있음
        if distances[optimal_idx_roc] < 0.8:
            thresholds_list.append(thresholds_roc[optimal_idx_roc])
        
        # 2. Youden's J statistic (민감도 + 특이도 - 1)
        j_statistic = tpr - fpr
        optimal_j_idx = np.argmax(j_statistic)
        thresholds_list.append(thresholds_roc[optimal_j_idx])
        
        # 3. F1 스코어 최대화 임계값
        f1_scores = []
        test_thresholds = np.linspace(0.1, 0.9, 9)  # 0.1부터 0.9까지 9개 지점 테스트
        
        for threshold in test_thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            # 모든 예측이 한 클래스인 경우 처리
            if len(np.unique(y_pred)) == 1:
                f1 = 0  # 한 클래스만 예측하면 F1 스코어가 의미 없음
            else:
                # class labels가 (0,1)인지 확인
                labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
                pos_label = 1 if 1 in labels else labels[-1]
                
                # 수동으로 f1 계산하여 제로 디비전 오류 방지
                try:
                    from sklearn.metrics import precision_score, recall_score
                    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
                    recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
                    
                    f1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
                except Exception:
                    f1 = 0
            f1_scores.append(f1)
        
        # F1 스코어가 최대인 임계값 (0일 수 있으므로 주의)
        if max(f1_scores) > 0:
            f1_optimal_threshold = test_thresholds[np.argmax(f1_scores)]
            thresholds_list.append(f1_optimal_threshold)
        
        # 4. 클래스 불균형을 고려한 임계값
        # 극단적인 불균형(15% 미만 또는 85% 초과)이 있으면 클래스 비율에 가까운 임계값도 고려
        if class_ratio < 0.15 or class_ratio > 0.85:
            balanced_threshold = (class_ratio + 0.5) / 2  # 클래스 비율과 0.5 사이의 중간값
            thresholds_list.append(balanced_threshold)
        
        # 5. 정확도 최대화 임계값
        accuracies = []
        for threshold in test_thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            accuracy = np.mean(y_pred == y_true)
            accuracies.append(accuracy)
        
        accuracy_optimal_threshold = test_thresholds[np.argmax(accuracies)]
        thresholds_list.append(accuracy_optimal_threshold)
        
        # 결과 임계값들의 중앙값 사용 (이상치에 강건함)
        if thresholds_list:
            final_threshold = np.median(thresholds_list)
            
            # 임계값이 너무 극단적이면 보정 (0.2~0.8 범위로 제한)
            final_threshold = max(0.2, min(0.8, final_threshold))
            
            # 임계값 유효성 검사 (NaN 체크)
            if np.isnan(final_threshold):
                return 0.5  # NaN인 경우 기본값 반환
                
            return final_threshold
        else:
            return 0.5  # 유효한 임계값을 찾지 못한 경우
            
    except Exception as e:
        try:
            import streamlit as st
            st.error(f"최적 임계값 찾기 오류: {str(e)}")
        except ImportError:
            print(f"최적 임계값 찾기 오류: {str(e)}")
            import traceback
            print(traceback.format_exc())
        return 0.5  # 오류 시 기본값 반환

def objective(trial, X_train, X_test, y_train, y_test, model_type, class_weight=None):
    """Optuna의 목적 함수: 모델 하이퍼파라미터를 튜닝합니다."""
    try:
        if model_type == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),  # 상한 증가
                'max_depth': trial.suggest_int('max_depth', 3, 15),  # 상한 증가
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),  # 추가
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),  # 추가
                'random_state': 42,
                'class_weight': class_weight
            }
            model = RandomForestClassifier(**params)

        elif model_type == 'GradientBoosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),  # 상한 증가
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),  # 하한 감소, 로그 스케일 사용
                'max_depth': trial.suggest_int('max_depth', 3, 15),  # 상한 증가
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'max_features': trial.suggest_float('max_features', 0.3, 1.0),  # 추가
                'random_state': 42
            }
            model = GradientBoostingClassifier(**params)
        
        else:
            raise ValueError(f"지원되지 않는 모델 유형: {model_type}")
        
        # 모델 훈련
        model.fit(X_train, y_train)
        
        # 성능 평가
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        return roc_auc
    except Exception as e:
        st.error(f"하이퍼파라미터 최적화 오류: {str(e)}")
        return 0.0  # 오류 시 최저 점수 반환

def save_complete_model(model, X, selected_features, params, results, category_dir):
    """모델과 관련 정보를 완전하게 저장하는 함수"""
    try:
        # 1. 모델 자체 저장
        model_path = os.path.join(category_dir, 'best_model.pkl')
        joblib.dump(model, model_path)
        #st.write(f"model saved: {model_path}")
        
        # 2. 모델 정보 저장 (모델과 관련된 모든 중요 정보 포함)
        model_info = {
            'model_type': type(model).__name__,
            'model_parameters': params,
            'features': X.columns.tolist(),  # 모든 사용된 특성 목록
            'selected_features': selected_features,  # 선택된 특성 목록
            'performance': results  # 모델 성능 정보
        }
        
        # 특성 중요도가 있는 경우 추가
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.Series(
                model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            model_info['feature_importance'] = feature_importance.to_dict()
        
        # 모델 정보를 별도 파일로 저장
        info_path = os.path.join(category_dir, 'model_info.pkl')
        joblib.dump(model_info, info_path)
        #st.write(f"Completed Saving Model Infomation: {info_path}")
        
        # 3. 읽기 쉬운 형식으로도 저장 (JSON)
        json_info = model_info.copy()
        # JSON 직렬화를 위해 복잡한 객체를 문자열로 변환
        json_info['model_parameters'] = {k: str(v) for k, v in params.items()}
        
        json_path = os.path.join(category_dir, 'model_info.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_info, f, indent=2, ensure_ascii=False)
        #st.write(f"⬇️Saved as JSON: {json_path}")
        
        return model_path, info_path
    except Exception as e:
        st.error(f"모델 저장 중 오류: {str(e)}")
        return None, None

def build_improved_prediction_model(model_data, category, output_dir, n_features=30, n_trials=50):
    """개선된 예측 모델을 구축합니다. 시각화는 생성하여 세션에 저장하지만 화면에는 표시하지 않습니다."""

   
    # 데이터 검증
    if category not in model_data:
        st.error(f"{category} 카테고리에 대한 모델 데이터가 없습니다.")
        return None
    
    df = model_data[category]
    
    # 데이터 크기 확인
    if len(df) < 30:
        st.error(f"{category} 카테고리의 데이터가 부족합니다 ({len(df)} rows). 모델링을 건너뜁니다.")
        return None
    
    # 타겟 변수 제외한 모든 특성 컬럼
    all_feature_columns = [col for col in df.columns if col != category]
    
    # 입력 검증
    if len(all_feature_columns) == 0:
        st.error(f"오류: 사용 가능한 변수가 없습니다. df 컬럼: {df.columns.tolist()}")
        return None

    # 특성 및 타겟 변수 분리
    X = df[all_feature_columns]
    
    # 방향 예측 (이진 분류: 0 또는 1)
    direction = np.where(df[category] > 0, 1, 0)
    
    # 클래스 불균형 확인
    class_count = np.bincount(direction)
    total_samples = len(direction)
    
    # 클래스 분포가 너무 극단적인지 확인
    class_ratio = min(class_count) / total_samples
    if class_ratio < 0.1:  # 10% 미만의 클래스가 있으면 경고
        st.warning(f"Significant class imbalance detected: the minority class represents just {class_ratio:.2%} of the data.")
    
    # 특성 선택 (너무 많은 특성은 과적합을 유발할 수 있음)
    try:
        selected_features = select_best_features(X, direction, all_feature_columns, n_features=n_features)
        
        # 선택된 특성만 사용
        X = X[selected_features]
    except Exception as e:
        st.error(f"특성 선택 중 오류: {str(e)}")
        # 오류 시 원래 특성 사용
        selected_features = all_feature_columns[:min(n_features, len(all_feature_columns))]
        X = X[selected_features]
        st.write(f"오류로 인해 첫 {len(selected_features)}개 특성을 사용합니다.")
    
    # 심각한 클래스 불균형이 있는 경우 클래스 가중치 적용
    class_weight = None
    if min(class_count) / total_samples < 0.3:  # 기준치 상향 조정
        # 기존 'balanced' 대신 명시적으로 down 클래스에 더 높은 가중치 부여
        down_class_idx = np.argmin(class_count)  # 소수 클래스 (아마도 'down')
        up_class_idx = 1 - down_class_idx        # 다수 클래스 (아마도 'up')
        
        # down 클래스에 2배 ~ 3배 가중치 (불균형 정도에 따라 조정)
        weight_ratio = max(2.0, min(3.0, 1 / (min(class_count) / total_samples)))
        
        class_weight = {down_class_idx: weight_ratio, up_class_idx: 1.0}
        st.info(f"Due to class imbalance, a weight of {weight_ratio:.2f}x is applied to class {down_class_idx}.")
    
    # 테스트 크기 비율 통일
    test_size_ratio = 0.15   # 20% 테스트, 70% 훈련
    
    # 학습/테스트 분할 (시계열 특성 유지)
    split_index = int(len(X) * (1 - test_size_ratio))
    if split_index <= 0 or split_index >= len(X):
        st.error(f"데이터 분할 오류: split_index={split_index}, 데이터 길이={len(X)}")
        return None
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = direction[:split_index], direction[split_index:]
    
    # 카테고리별 디렉토리 생성
    category_dir = os.path.join(output_dir, 'categories', category)
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)
    
    model_types = ["RandomForest", "GradientBoosting"]

    model_results = {}
    best_auc = 0
    best_model_name = None
    best_model = None
    
    # 각 모델 유형에 대해 Optuna 튜닝 수행
    for model_idx, model_type in enumerate(model_types):
        try:
            # Optuna 연구 객체 생성 (maximize로 수정)
            study = optuna.create_study(direction='maximize')
            
            # 최적화 실행
            study.optimize(
                lambda trial: objective(
                    trial, X_train, X_test, y_train, y_test,
                    model_type, class_weight
                ), 
                n_trials=n_trials
            )
        
            # 최적 모델 재구성
            if model_type == 'RandomForest':
                best_params = study.best_params.copy()
                best_params['class_weight'] = class_weight
                best_params['random_state'] = 42
                tuned_model = RandomForestClassifier(**best_params)
            else:  # GradientBoosting
                best_params = study.best_params.copy()
                best_params['random_state'] = 42
                tuned_model = GradientBoostingClassifier(**best_params)
            
            # 모델 훈련
            tuned_model.fit(X_train, y_train)
            # 예측 전 X_test의 NaN 검증 및 처리
            has_nan_before = np.isnan(X_test.values).any()
            if has_nan_before:
                st.warning(f"{model_type} 모델 예측 전 X_test에 NaN이 있습니다. 이를 처리합니다.")
                # 각 열의 중앙값으로 NaN 대체
                X_test = X_test.fillna(X_train.median())
                st.info(f"NaN을 각 특성의 중앙값으로 대체했습니다.")

            # 이제 NaN 없는 상태로 예측
            y_pred = tuned_model.predict(X_test)
            y_prob = tuned_model.predict_proba(X_test)[:, 1]

            if np.isnan(y_pred).all():
                st.error(f"{category} 예측 결과가 모두 NaN입니다. 모델을 저장하지 않습니다.")
                return None
            # ✅ 4. 예측 편향 체크
            if len(np.unique(y_pred)) <= 1:
                st.warning(f"The predictions for {category} are overly biased toward one class.")
            # 예측 결과 확인 (디버깅용)
            if np.isnan(y_pred).any():
                st.error(f"X_test의 NaN을 처리했음에도 y_pred에 NaN이 있습니다. 모델 자체에 문제가 있을 수 있습니다.")
                print(f"X_test had NaN before: {has_nan_before}")
                print(f"X_test NaN count after: {np.isnan(X_test.values).sum()}")
                print(f"y_pred NaN count: {np.isnan(y_pred).sum()}")
                print(f"y_pred unique values: {np.unique(y_pred)}")
                
                # NaN 위치 찾기
                nan_indices = np.where(np.isnan(y_pred))[0]
                if len(nan_indices) > 0:
                    print(f"NaN indices in y_pred: {nan_indices[:5]}")
                    # 해당 행 살펴보기
                    for idx in nan_indices[:2]:  # 처음 2개만 출력
                        print(f"X_test row {idx}:")
                        print(X_test.iloc[idx])

            # 성능 평가
            accuracy = accuracy_score(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # 결과 저장
            model_results[model_type] = {
                "model": tuned_model,
                "accuracy": accuracy,
                "roc_auc": roc_auc,
                "params": best_params,
                "y_pred": y_pred,
                "y_prob": y_prob
            }
            
            # 최고 성능 모델 업데이트
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_model_name = model_type
                best_model = tuned_model
        
        except Exception as e:
            st.error(f"{model_type} 모델 최적화 중 오류: {str(e)}")
    
    # 최고 성능 모델이 없는 경우
    if best_model is None or best_model_name is None:
        st.error("모든 모델 학습에 실패했습니다.")
        return None
    
    # 최고 성능 모델 관련 변수 설정
    best_params = model_results[best_model_name]["params"]
    y_pred = model_results[best_model_name]["y_pred"]
    y_prob = model_results[best_model_name]["y_prob"]
    
    # 하이퍼파라미터 저장
    with open(os.path.join(category_dir, 'best_params.txt'), 'w') as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
    
    # 성능 평가
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    
    # 특성 중요도
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.Series(
            best_model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
    else:
        feature_importance = pd.Series(
            [0] * len(X.columns),
            index=X.columns
        )
    
    # 최적 임계값 찾기
    optimal_threshold = find_optimal_threshold(y_test, y_prob)
    if np.isnan(optimal_threshold):
        st.warning(f"최적 임계값이 NaN입니다. 기본값 0.5를 사용합니다.")
        optimal_threshold = 0.5
    if np.allclose(y_prob, y_prob[0]):
        st.warning(f"{category}의 예측 확률이 모두 유사합니다. 모델 불안정 가능성 있음.")
        # 클래스 분포 확인
    up_ratio = np.mean(y_test)
    down_ratio = 1 - up_ratio
    # 'up' 예측이 과도하게 많은지 확인
    y_pred_default = (y_prob >= 0.5).astype(int)
    pred_up_ratio = np.mean(y_pred_default)
    if pred_up_ratio > up_ratio + 0.1:  # 실제보다 10% 이상 많으면
            
        # 임계값 상향 조정 (up 예측을 줄이기 위해)
        adjusted_threshold = max(optimal_threshold + 0.1, 0.6)
        st.write(f"The model is overpredicting 'up' (Predicted: {pred_up_ratio:.2f}, Actual: {up_ratio:.2f}). Adjusting the threshold to {adjusted_threshold:.2f}.")
        optimal_threshold = min(adjusted_threshold, 0.8)  # 최대 0.8로 제한


    # 최적 임계값 기준 예측 
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

    # NaN 체크 및 처리
    if np.isnan(y_pred_optimal).any():
        st.warning(f"y_pred_optimal에 NaN 값이 있습니다. 대체 처리합니다.")
        # 가장 많은 클래스로 대체
        most_common_class = np.bincount(y_train).argmax()
        y_pred_optimal = np.nan_to_num(y_pred_optimal, nan=most_common_class)

    # 항상 정확도 계산 (NaN 값의 유무와 관계없이)
    accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
    # 조정 후 확인
    adjusted_up_ratio = np.mean(y_pred_optimal)
    st.write(f"'Up' prediction ratio before adjustment: {pred_up_ratio:.2f}, after adjustment: {adjusted_up_ratio:.2f}, actual: {up_ratio:.2f}.")

   
    print("y_pred_optimal values:", np.unique(y_pred_optimal))
    

    # 성능 결과 모음
    performance_results = {
        "accuracy": float(accuracy),
        "roc_auc": float(best_auc),
        "optimal_threshold": float(optimal_threshold),
        "accuracy_optimal": float(accuracy_optimal),
        "classification_report": classification_rep
    }

    # 향상된 모델 저장 함수 호출
    model_path, info_path = save_complete_model(
        model=best_model,
        X=X,
        selected_features=selected_features,
        params=best_params,
        results=performance_results,
        category_dir=category_dir
    )
    
    # 세션 상태에 시각화 저장소 초기화
    if 'visualizations' not in st.session_state:
        st.session_state['visualizations'] = {}
    
    # 시각화 섹션 추가 - 화면에 표시하지 않고 세션에만 저장
    # 특성 중요도 시각화
    try:
        feature_importance_fig, ax = plt.subplots(figsize=(10, 8))
        top_features = feature_importance.head(min(15, len(feature_importance)))
        top_features.plot(kind='barh', color='teal', ax=ax)
        
        total = top_features.sum()
        for i, v in enumerate(top_features):
            ax.text(v + 0.01, i, f'{v:.3f} ({v/total * 100:.1f}%)', color='black', va='center', fontweight='bold')
        
        plt.title(f'{category} - Feature Importance', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        
        # 세션 상태에 시각화 저장
        st.session_state['visualizations'][f"{category}_feature_importance"] = feature_importance_fig
        
        # 파일로 저장
        plt.savefig(os.path.join(category_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        st.error(f"특성 중요도 시각화 오류: {str(e)}")
    
    # 혼동 행렬 (Confusion Matrix)
    try:
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrix_fig, ax = plt.subplots(figsize=(10, 8))
        # 백분율로 변환
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        # 혼동 행렬 시각화 (개수와 백분율 함께 표시)
        sns.heatmap(cm, annot=np.asarray([
            [f'{cm[0,0]} ({cm_percent[0,0]:.1f}%)', f'{cm[0,1]} ({cm_percent[0,1]:.1f}%)'],
            [f'{cm[1,0]} ({cm_percent[1,0]:.1f}%)', f'{cm[1,1]} ({cm_percent[1,1]:.1f}%)']
        ]), fmt='', cmap='Blues', cbar=False, annot_kws={"size": 14}, ax=ax)
        
        plt.title(f'Direction Prediction Confusion Matrix\nAccuracy: {accuracy:.2%}', fontsize=16)
        plt.xlabel('Predicted Value', fontsize=14)
        plt.ylabel('Actual Value', fontsize=14)
        plt.xticks([0.5, 1.5], ['Down', 'Up'], fontsize=12)
        plt.yticks([0.5, 1.5], ['Down', 'Up'], fontsize=12)
        plt.tight_layout()
        
        # 세션 상태에 시각화 저장
        st.session_state['visualizations'][f"{category}_Confusion Matrix"] = confusion_matrix_fig
        
        # 파일로 저장
        plt.savefig(os.path.join(category_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        st.error(f"혼동 행렬 시각화 오류: {str(e)}")
    
    # ROC 곡선
    try:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        roc_curve_fig, ax = plt.subplots(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Predictions')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'ROC Curve', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 세션 상태에 시각화 저장
        st.session_state['visualizations'][f"{category}roc_curve"] = roc_curve_fig
        
        # 파일로 저장
        plt.savefig(os.path.join(category_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        st.error(f"ROC 곡선 시각화 오류: {str(e)}")
    
    # 클래스별 예측 확률 분포
    try:
        probability_distribution_fig, ax = plt.subplots(figsize=(10, 8))
        sns.histplot(y_prob[y_test == 0], bins=20, alpha=0.5, color='red', label='Actual Down (0)', ax=ax)
        sns.histplot(y_prob[y_test == 1], bins=20, alpha=0.5, color='green', label='Actual Up (1)', ax=ax)
        
        plt.axvline(x=0.5, color='gray', linestyle='--', label='Standard threshold (0.5)')
        plt.axvline(x=optimal_threshold, color='black', linestyle='--', label=f'Optimal threshold ({optimal_threshold:.3f})')
        plt.xlabel('Up Class[1] Prediction Probability', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title(f'Prediction Probability Distributions', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 세션 상태에 시각화 저장
        st.session_state['visualizations'][f"{category}probability_distribution"] = probability_distribution_fig
        
        # 파일로 저장
        plt.savefig(os.path.join(category_dir, 'probability_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        st.error(f"확률 분포 시각화 오류: {str(e)}")
   
    # 시간적 패턴 분석
    if hasattr(df, 'index') and len(df.index) == len(y_test) + len(y_train):
        try:
            prediction_trends_fig, ax = plt.subplots(figsize=(10, 8))
            
            # 테스트 데이터의 인덱스 가져오기
            test_indices = df.index[-len(y_test):]
            
            # 예측 결과와 실제 값 시각화
            plt.plot(test_indices, y_test, 'bo-', label='Actual direction', alpha=0.6, markersize=8)
            plt.plot(test_indices, y_pred, 'ro-', label='Predicted direction', alpha=0.6, markersize=8)
            
            # 오분류 포인트 강조
            incorrect_mask = y_test != y_pred
            incorrect_idx = test_indices[incorrect_mask]
            incorrect_actual = y_test[incorrect_mask]
            plt.scatter(incorrect_idx, incorrect_actual, color='yellow', s=150, zorder=5, 
                        label='Misclassification', edgecolors='black')
            
            plt.title(f'Predicted vs actual direction over time', fontsize=16)
            plt.xlabel('Time', fontsize=14)
            plt.ylabel('Direction (0=Down, 1=UP)', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 세션 상태에 시각화 저장
            st.session_state['visualizations'][f"{category}temporal_prediction"] = prediction_trends_fig
            
            # 파일로 저장
            plt.savefig(os.path.join(category_dir, 'temporal_prediction.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            st.error(f"시간적 패턴 시각화 오류: {str(e)}")
    
    # 분류 보고서 표시
    try:
        class_report_df = pd.DataFrame(classification_rep).transpose()
    except Exception as e:
        st.error(f"Error Displaying Classification Report: {str(e)}")
    
    # 결과 저장
    result = {
        'model_name': best_model_name,
        'accuracy': float(accuracy),
        'accuracy_optimal': float(accuracy_optimal),
        'optimal_threshold': float(optimal_threshold),
        'classification_report': classification_rep,
        'feature_importance': feature_importance.to_dict() if hasattr(best_model, 'feature_importances_') else {},
        'roc_auc': float(roc_auc),
        'model_path': os.path.join(category_dir, 'best_model.pkl'),
        'selected_features': selected_features
    }
    
    # 결과를 JSON으로 저장
    try:
        with open(os.path.join(category_dir, 'model_results.json'), 'w') as f:
            json.dump(result, f, indent=4)
    except Exception as e:
        st.error(f"결과 저장 오류: {str(e)}")
    
    # 모델 메타데이터 생성 및 저장
    try:
        model_metadata = {
            'execution_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'category': category,
            'data_size': {
                'total_samples': int(len(direction)),
                'train_samples': int(len(y_train)),
                'test_samples': int(len(y_test)),
                'features_count': int(len(selected_features))
            },
            'features': {
                'all_features': X.columns.tolist(),
                'top_importance_features': feature_importance.head(min(10, len(feature_importance))).index.tolist()
            },
            'class_distribution': {
                'train': {
                    'up': int(sum(y_train)),
                    'down': int(len(y_train) - sum(y_train)),
                    'up_ratio': float(sum(y_train) / len(y_train))
                },
                'test': {
                    'up': int(sum(y_test)),
                    'down': int(len(y_test) - sum(y_test)),
                    'up_ratio': float(sum(y_test) / len(y_test))
                }
            },
            'performance_summary': {
                'accuracy': float(accuracy),
                'accuracy_optimal': float(accuracy_optimal),
                'optimal_threshold': float(optimal_threshold),
                'roc_auc': float(roc_auc),
                'f1_score': float(classification_rep['1']['f1-score']),  # 상승 클래스의 F1 점수
            },
            'model_info': {
                'type': best_model_name,
                'parameters': str(best_model.get_params())
            },
            'files': {
                'model_path': 'best_model.pkl',
                'performance_plots': [
                    'confusion_matrix.png',
                    'roc_curve.png',
                    'feature_importance.png',
                    'probability_distribution.png'
                ]
            }
        }
        
        # 시간적 특성 플롯이 생성된 경우 추가 
        if hasattr(df, 'index') and len(df.index) == len(y_test) + len(y_train):
            model_metadata['files']['performance_plots'].append('temporal_prediction.png')
        
        # 메타데이터를 JSON으로 저장
        with open(os.path.join(category_dir, 'model_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(model_metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Error saving metadata: {str(e)}")
    
    # 예측 결과를 모델 데이터에 저장
    try:
        # 원본 데이터프레임 복사
        df_with_predictions = df.copy()
        if np.isnan(y_pred_optimal).any():
            st.warning("예측 결과에 NaN 값이 있습니다. 대체 처리합니다.")
            y_pred_optimal = np.nan_to_num(y_pred_optimal, nan=0)
        # 테스트 인덱스에 해당하는 행에 예측 결과 추가
        df_with_predictions.loc[X_test.index, 'predictions'] = y_pred_optimal
        
        # 업데이트된 데이터프레임을 모델 데이터에 저장 (모델 데이터를 직접 업데이트)
        model_data[category] = df_with_predictions
        if 'model_data' not in st.session_state:
                st.session_state['model_data'] = {}
                st.session_state['model_data'][category] = df_with_predictions

        if 'enhanced_model_data' not in st.session_state:
            st.session_state['enhanced_model_data'] = {}

        st.session_state['enhanced_model_data'][category] = df_with_predictions
        
        # 예측 결과에 대한 정보 추가
        result['test_indices'] = X_test.index.tolist()
        result['predictions'] = y_pred_optimal.tolist()
        
        # 간단한 성공 메시지만 표시
        st.success(f"✅ {category} Model building for each category is complete! Please check the results in each tab.")
    except Exception as e:
        st.warning(f"예측 결과 저장 중 오류: {str(e)}")
    
    return result



def analyze_data_quality(daily_avg_prices, daily_indicators):
    """Analyze data quality and display basic information."""
    st.subheader("Data Quality Analysis")

    # Data validation
    if daily_avg_prices is None or daily_indicators is None:
        st.error("Missing required data for quality analysis.")
        return

    # Analyze price data
    st.write("### Price Data Overview:")
    st.write(f"Date Range: {daily_avg_prices.index.min()} ~ {daily_avg_prices.index.max()}")
    #st.write(f"Number of Categories: {daily_avg_prices.shape[1]}")
    #st.write(f"Number of Dates: {daily_avg_prices.shape[0]}")
    st.write(f"Number of Missing Values (NaN): {daily_avg_prices.isna().sum().sum()}")

    # Basic statistics per category
    st.write("### Basic Statistics by Category:")
    stats_df = pd.DataFrame()

    for category in daily_avg_prices.columns:
        data = daily_avg_prices[category]
        stats = {
            "Count": data.count(),
            "Mean": data.mean(),
            "Standard Deviation": data.std(),
            "Min": data.min(),
            "Max": data.max(),
            "Coefficient of Variation": data.std() / data.mean() if data.mean() != 0 else float('nan')
        }
        stats_df[category] = pd.Series(stats)

    st.dataframe(stats_df.T)

    # Analyze indicator data
    st.write("### Indicator Data Overview:")
    st.write(f"Date Range: {daily_indicators.index.min()} ~ {daily_indicators.index.max()}")
    st.write(f"Number of Indicators: {daily_indicators.shape[1]}")
    st.write(f"Number of Dates: {daily_indicators.shape[0]}")
    st.write(f"Number of Missing Values (NaN): {daily_indicators.isna().sum().sum()}")

    # Basic statistics for indicators
    if daily_indicators.shape[1] > 0:
        st.write("### Basic Statistics by Indicator:")
        indicator_stats = daily_indicators.describe().T
        st.dataframe(indicator_stats)

    # Check for date alignment
    common_dates = daily_avg_prices.index.intersection(daily_indicators.index)
    st.write(f"Number of matching dates between price and indicator data: {len(common_dates)}")

    # Check normalized variables
    norm_cols = [col for col in daily_indicators.columns if '_norm' in col]
    st.write(f"Number of normalized indicators: {len(norm_cols)}")

    # Correlation heatmap between top indicators and categories
    st.write("### Correlation between Key Indicators:")
    
    # 상위 경제 지표와 모든 카테고리 선택
    try:
        top_indicators = daily_indicators.columns[:min(5, len(daily_indicators.columns))]  # 상위 5개 지표
        correlation_data = pd.concat([daily_avg_prices, daily_indicators[top_indicators]], axis=1)
        
        # 상관관계 히트맵
        fig, ax = plt.subplots(figsize=(10, 7))
        correlation_matrix = correlation_data.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', 
                    linewidths=0.5, vmin=-1, vmax=1, center=0, ax=ax)
        plt.title('Correlation between key Indicators', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        st.success("""
        ### 🧠 Kernaussagen:

        - 🛡️ **Defense & Gold:** Unsicherheit treibt Gold und Rüstungsaktien nach oben.
        - 💻 **Tech & Dollar:** Starker Dollar schwächt Tech-Aktien.
        - 📉 **Fed Rate:** Höhere Zinsen bremsen Wirtschaft und Konsum.
        """)
        plt.close()
        

    except Exception as e:
        st.error(f"상관관계 히트맵 생성 오류: {str(e)}")
    
    # 시계열 데이터 시각화
    #st.write("### Price trends by category:")
    # 누적 수익률 계산
    try:
        daily_returns = daily_avg_prices.pct_change()  # 일간 수익률
        cumulative_returns = (1 + daily_returns).cumprod()  # 누적 수익률
    except Exception as e:
        st.error(f"누적 수익률 계산 오류: {str(e)}")
        cumulative_returns = None

    # 시계열 시각화
    if cumulative_returns is not None:
        #st.write("### Cumulative Returns by Category")
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            for category in cumulative_returns.columns:
                ax.plot(
                    cumulative_returns.index,
                    cumulative_returns[category],
                    label=category,
                    linewidth=2
                )
            ax.set_title('Cumulative Return by Category', fontsize=16)
            ax.set_xlabel('Date', fontsize=14)
            ax.set_ylabel('Cumulative Return', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            #st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"누적 수익률 시각화 오류: {str(e)}")

def detect_extreme_events(df, daily_avg_prices, daily_indicators):
    """
    경제 지표, 가격 및 감성 지표의 극단적 이벤트를 감지합니다.
    
    Parameters:
    -----------
    df : DataFrame
        원본 데이터프레임
    daily_avg_prices : DataFrame
        일별 평균 가격 데이터
    daily_indicators : DataFrame
        일별 지표 데이터
    
    Returns:
    --------
    extreme_events_df : DataFrame
        감지된 모든 극단적 이벤트를 포함하는 데이터프레임
    """
    # 이벤트 저장을 위한 리스트
    extreme_events = []
    
    # 데이터 유효성 확인 - 가장 먼저 체크
    if df is None or df.empty:
        st.warning("No data for analysis")
        return pd.DataFrame(columns=['date', 'event_type', 'indicator', 'value', 
                                   'threshold', 'percentile', 'description'])
    
    if daily_indicators is None or daily_indicators.empty:
        st.warning("No indicators data")
        return pd.DataFrame(columns=['date', 'event_type', 'indicator', 'value', 
                                   'threshold', 'percentile', 'description'])
    
    if daily_avg_prices is None or daily_avg_prices.empty:
        st.warning("No price data")
        return pd.DataFrame(columns=['date', 'event_type', 'indicator', 'value', 
                                   'threshold', 'percentile', 'description'])
    
    # ✅ 안전한 datetime 변환
    try:
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        if not pd.api.types.is_datetime64_any_dtype(daily_indicators.index):
            daily_indicators.index = pd.to_datetime(daily_indicators.index, errors='coerce')
        
        if not pd.api.types.is_datetime64_any_dtype(daily_avg_prices.index):
            daily_avg_prices.index = pd.to_datetime(daily_avg_prices.index, errors='coerce')
    except Exception as e:
        st.error(f"날짜 변환 오류: {str(e)}")
        return pd.DataFrame(columns=['date', 'event_type', 'indicator', 'value', 
                                   'threshold', 'percentile', 'description'])
    
    # 1. 경제 지표 극단값 (상위/하위 10%)
    try:
        economic_columns = [col for col in daily_indicators.columns if any(
            econ in col for econ in [
                'GDP', 'CPI', 'Dollar_Index', 'Fed_Funds_Rate', 
                'Gov_Spending', 'Industrial_Production', 'Natural_Gas',
                'Real_Interest_Rate', 'WTI_Oil', '10Y_Treasury_Yield'
            ]
        )]
        
        for col in economic_columns:
            try:
                # 상위 10% 임계값
                high_threshold = daily_indicators[col].quantile(0.90)
                # 하위 10% 임계값
                low_threshold = daily_indicators[col].quantile(0.10)
                
                # 상위 10% 이벤트 감지
                high_extreme_dates = daily_indicators[daily_indicators[col] > high_threshold].index
                for date in high_extreme_dates:
                    extreme_events.append({
                        'date': date,
                        'event_type': 'Extreme High',
                        'indicator': col,
                        'value': daily_indicators.loc[date, col],
                        'threshold': high_threshold,
                        'percentile': 90,
                        'description': f'High {col.split("_")[0]} ({daily_indicators.loc[date, col]:.2f})'
                    })
                
                # 하위 10% 이벤트 감지
                low_extreme_dates = daily_indicators[daily_indicators[col] < low_threshold].index
                for date in low_extreme_dates:
                    extreme_events.append({
                        'date': date,
                        'event_type': 'Extreme Low',
                        'indicator': col,
                        'value': daily_indicators.loc[date, col],
                        'threshold': low_threshold,
                        'percentile': 10,
                        'description': f'Low {col.split("_")[0]} ({daily_indicators.loc[date, col]:.2f})'
                    })
            except Exception as e:
                st.warning(f"{col} 지표 분석 중 오류: {str(e)}")
                continue
    except Exception as e:
        st.warning(f"경제 지표 분석 중 오류: {str(e)}")
    
    # 2. 가격 급등락 감지 (5% 이상 변화)
    try:
        for category in daily_avg_prices.columns:
            try:
                # 일일 변화율 계산 (퍼센트)
                daily_returns = daily_avg_prices[category].pct_change() * 100
                daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
                
                # 급등 감지 (5% 이상 상승)
                surge_dates = daily_returns[daily_returns > 5].index
                for date in surge_dates:
                    extreme_events.append({
                        'date': date,
                        'event_type': 'Price Surge',
                        'indicator': f'{category}_price',
                        'value': daily_returns.loc[date],
                        'threshold': 5,
                        'percentile': None,
                        'description': f'{category} Surge (+{daily_returns.loc[date]:.2f}%)'
                    })
                
                # 급락 감지 (5% 이상 하락)
                crash_dates = daily_returns[daily_returns < -5].index
                for date in crash_dates:
                    extreme_events.append({
                        'date': date,
                        'event_type': 'Price Crash',
                        'indicator': f'{category}_price',
                        'value': daily_returns.loc[date],
                        'threshold': -5,
                        'percentile': None,
                        'description': f'{category} Crash ({daily_returns.loc[date]:.2f}%)'
                    })
            except Exception as e:
                st.warning(f"{category} 가격 분석 중 오류: {str(e)}")
                continue
    except Exception as e:
        st.warning(f"가격 분석 중 오류: {str(e)}")
    
    # 3. 극단적 감성 변화 감지
    try:
        sentiment_cols = [col for col in daily_indicators.columns if any(
            sent in col for sent in [
                'sentiment_score', 'fear_greed', 'positive_prob', 
                'negative_prob', 'sentiment_group_Zscore'
            ]
        )]
        
        from scipy import stats
        for col in sentiment_cols:
            try:
                # 데이터가 충분한지 확인
                if daily_indicators[col].dropna().shape[0] < 2:
                    continue
                
                # Z-score 계산
                z_scores = stats.zscore(daily_indicators[col].dropna())
                z_df = pd.DataFrame({'z_score': z_scores}, index=daily_indicators[col].dropna().index)
                
                # 매우 긍정적인 감성 (Z-score > 2)
                very_positive_dates = z_df[z_df['z_score'] > 2].index
                for date in very_positive_dates:
                    extreme_events.append({
                        'date': date,
                        'event_type': 'Extreme Positive Sentiment',
                        'indicator': col,
                        'value': daily_indicators.loc[date, col],
                        'threshold': None,
                        'percentile': None,
                        'description': f'Extreme Positive Sentiment {col} (Z-score > 2)'
                    })
                
                # 매우 부정적인 감성 (Z-score < -2)
                very_negative_dates = z_df[z_df['z_score'] < -2].index
                for date in very_negative_dates:
                    extreme_events.append({
                        'date': date,
                        'event_type': 'Extreme Negative Sentiment',
                        'indicator': col,
                        'value': daily_indicators.loc[date, col],
                        'threshold': None,
                        'percentile': None,
                        'description': f'Extreme Negative Sentiment{col} (Z-score < -2)'
                    })
            except Exception as e:
                st.warning(f"{col} 감성 지표 분석 중 오류: {str(e)}")
                continue
    except Exception as e:
        st.warning(f"감성 지표 분석 중 오류: {str(e)}")
    
    # 데이터프레임으로 변환
    if extreme_events:
        try:
            extreme_events_df = pd.DataFrame(extreme_events)
            # 날짜 기준으로 정렬
            extreme_events_df = extreme_events_df.sort_values('date', ascending=False)
            return extreme_events_df
        except Exception as e:
            st.error(f"데이터프레임 생성 중 오류: {str(e)}")
            return pd.DataFrame(columns=['date', 'event_type', 'indicator', 'value', 
                                       'threshold', 'percentile', 'description'])
    else:
        return pd.DataFrame(columns=['date', 'event_type', 'indicator', 'value', 
                                   'threshold', 'percentile', 'description'])


def detect_technical_patterns(daily_avg_prices, window=20):
    """
    주요 기술적 패턴을 감지합니다 (골든 크로스, 데스 크로스, 추세 반전 등).
    
    Parameters:
    -----------
    daily_avg_prices : DataFrame
        일별 평균 가격 데이터
    window : int
        이동평균 윈도우 크기
    
    Returns:
    --------
    patterns_df : DataFrame
        감지된 기술적 패턴을 포함하는 데이터프레임
    """
    patterns = []
    
    if daily_avg_prices is None or daily_avg_prices.empty:
        return pd.DataFrame(columns=['date', 'category', 'pattern', 'description'])
    
    # datetime 형식 변환
    try:
        if not pd.api.types.is_datetime64_any_dtype(daily_avg_prices.index):
            daily_avg_prices.index = pd.to_datetime(daily_avg_prices.index, errors='coerce')
    except Exception as e:
        st.error(f"날짜 변환 오류: {str(e)}")
        return pd.DataFrame(columns=['date', 'category', 'pattern', 'description'])
    
    for category in daily_avg_prices.columns:
        try:
            prices = daily_avg_prices[category]
            
            # NaN 값 확인
            if prices.isnull().all():
                continue
            
            # 이동평균 계산
            ma_short = prices.rolling(window=5).mean()  # 5일 이동평균
            ma_medium = prices.rolling(window=20).mean()  # 20일 이동평균
            ma_long = prices.rolling(window=50).mean()  # 50일 이동평균
            
            # 전일 이동평균
            ma_short_prev = ma_short.shift(1)
            ma_medium_prev = ma_medium.shift(1)
            ma_long_prev = ma_long.shift(1)
            
            try:
                # 1. 골든 크로스 감지
                golden_cross_5_20 = (ma_short > ma_medium) & (ma_short_prev <= ma_medium_prev)
                golden_cross_dates_5_20 = golden_cross_5_20[golden_cross_5_20].index
                
                for date in golden_cross_dates_5_20:
                    patterns.append({
                        'date': date,
                        'category': category,
                        'pattern': 'Golden Cross (5-20)',
                        'description': f'{category}: 5 MA crossed above 20 MA'
                    })
                
                golden_cross_20_50 = (ma_medium > ma_long) & (ma_medium_prev <= ma_long_prev)
                golden_cross_dates_20_50 = golden_cross_20_50[golden_cross_20_50].index
                
                for date in golden_cross_dates_20_50:
                    patterns.append({
                        'date': date,
                        'category': category,
                        'pattern': 'Golden Cross (20-50)',
                        'description': f'{category}: 20 MA crossed above 50 MA'
                    })
            except Exception as e:
                st.warning(f"{category} 골든 크로스 감지 중 오류: {str(e)}")
            
            try:
                # 2. 데스 크로스 감지
                death_cross_5_20 = (ma_short < ma_medium) & (ma_short_prev >= ma_medium_prev)
                death_cross_dates_5_20 = death_cross_5_20[death_cross_5_20].index
                
                for date in death_cross_dates_5_20:
                    patterns.append({
                        'date': date,
                        'category': category,
                        'pattern': 'Death Cross (5-20)',
                        'description': f'{category}: 5 MA crossed below 20 MA'
                    })
                
                death_cross_20_50 = (ma_medium < ma_long) & (ma_medium_prev >= ma_long_prev)
                death_cross_dates_20_50 = death_cross_20_50[death_cross_20_50].index
                
                for date in death_cross_dates_20_50:
                    patterns.append({
                        'date': date,
                        'category': category,
                        'pattern': 'Death Cross (20-50)',
                        'description': f'{category}: 20 MA crossed below 50 MA'
                    })
            except Exception as e:
                st.warning(f"{category} 데스 크로스 감지 중 오류: {str(e)}")
            
            try:
                # 3. 추세 반전 감지
                uptrend_reversal = (prices < ma_medium) & (prices.shift(1) >= ma_medium_prev)
                uptrend_reversal_dates = uptrend_reversal[uptrend_reversal].index
                
                for date in uptrend_reversal_dates:
                    patterns.append({
                        'date': date,
                        'category': category,
                        'pattern': 'Uptrend Reversal',
                        'description': f'{category}:Price dropped below 20 MA'
                    })
                
                downtrend_reversal = (prices > ma_medium) & (prices.shift(1) <= ma_medium_prev)
                downtrend_reversal_dates = downtrend_reversal[downtrend_reversal].index
                
                for date in downtrend_reversal_dates:
                    patterns.append({
                        'date': date,
                        'category': category,
                        'pattern': 'Downtrend Reversal',
                        'description': f'{category}:Price rosw above 20 MA'
                    })
            except Exception as e:
                st.warning(f"{category} 추세 반전 감지 중 오류: {str(e)}")
            
            try:
                # 4. 지지선/저항선 테스트
                support_test = (
                    (prices <= ma_long * 1.005) &  
                    (prices > ma_long) &  
                    (prices.shift(1) > ma_long * 1.005)
                )
                support_test_dates = support_test[support_test].index
                
                for date in support_test_dates:
                    patterns.append({
                        'date': date,
                        'category': category,
                        'pattern': 'Support Test',
                        'description': f'{category}: Supported at 50 MA'
                    })
                
                resistance_test = (
                    (prices >= ma_long * 0.995) &  
                    (prices < ma_long) &  
                    (prices.shift(1) < ma_long * 0.995)
                )
                resistance_test_dates = resistance_test[resistance_test].index
                
                for date in resistance_test_dates:
                    patterns.append({
                        'date': date,
                        'category': category,
                        'pattern': 'Resistance Test',
                        'description': f'{category}: Resistance at 50 MA'
                    })
            except Exception as e:
                st.warning(f"{category} 지지선/저항선 테스트 감지 중 오류: {str(e)}")
        except Exception as e:
            st.warning(f"{category} 패턴 분석 중 오류: {str(e)}")
            continue
    
    # 데이터프레임으로 변환
    if patterns:
        try:
            patterns_df = pd.DataFrame(patterns)
            # 날짜 기준으로 정렬
            patterns_df = patterns_df.sort_values('date', ascending=False)
            return patterns_df
        except Exception as e:
            st.error(f"패턴 데이터프레임 생성 중 오류: {str(e)}")
            return pd.DataFrame(columns=['date', 'category', 'pattern', 'description'])
    else:
        return pd.DataFrame(columns=['date', 'category', 'pattern', 'description'])

def detect_black_swan_events(df, daily_avg_prices, daily_indicators, std_threshold=3.0):
    """
    블랙 스완 이벤트를 감지합니다 (매우 희귀하고 예측하기 어려운 극단적 이벤트).
    
    Parameters:
    -----------
    df : DataFrame
        원본 데이터프레임
    daily_avg_prices : DataFrame
        일별 평균 가격 데이터
    daily_indicators : DataFrame
        일별 지표 데이터
    std_threshold : float
        표준 편차 기준 임계값 (기본값: 3.0)
    
    Returns:
    --------
    black_swan_df : DataFrame
        감지된 블랙 스완 이벤트를 포함하는 데이터프레임
    """
    black_swan_events = []
    
    # 데이터 유효성 체크 먼저
    if df is None or df.empty:
        print("원본 데이터프레임이 비어 있습니다.")
        return pd.DataFrame(columns=['date', 'category', 'event_type', 'z_score', 'return_pct', 'description'])
    
    if daily_avg_prices is None or daily_avg_prices.empty:
        print("일별 평균 가격 데이터가 비어 있습니다.")
        return pd.DataFrame(columns=['date', 'category', 'event_type', 'z_score', 'return_pct', 'description'])
    
    # ✅ 날짜 형식 강제 변환
    try:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if daily_avg_prices is not None and not pd.api.types.is_datetime64_any_dtype(daily_avg_prices.index):
            daily_avg_prices.index = pd.to_datetime(daily_avg_prices.index, errors='coerce')
        if daily_indicators is not None and not pd.api.types.is_datetime64_any_dtype(daily_indicators.index):
            daily_indicators.index = pd.to_datetime(daily_indicators.index, errors='coerce')
    except Exception as e:
        print(f"날짜 변환 중 오류 발생: {str(e)}")
        return pd.DataFrame(columns=['date', 'category', 'event_type', 'z_score', 'return_pct', 'description'])
    
    # 1. 카테고리별 수익률 블랙 스완 감지
    for category in daily_avg_prices.columns:
        try:
            print(f"{category} Category Analysing...")
            
            # 로그 수익률 계산 (로그 계산을 위해 0과 음수 처리)
            price_data = daily_avg_prices[category].replace(0, np.nan) #0값처리 
            price_data = price_data.where(price_data > 0, np.nan) # 음수처리
            
            # 유효한 데이터 체크
            if price_data.isnull().all():
                print(f"{category} 카테고리는 유효한 데이터가 없습니다.")
                continue
                
            if len(price_data.dropna()) < 60:
                print(f"{category} 카테고리는 데이터가 부족합니다: {len(price_data.dropna())} < 60")
                continue
            
            # 벡터화된 방식으로 로그 수익률 계산
            shifted_price = price_data.shift(1)
            valid_pairs = (price_data > 0) & (shifted_price > 0)
            
            # 로그 수익률 계산 - 벡터화된 방식
            log_returns = pd.Series(np.nan, index=price_data.index, dtype='float64')
            log_returns[valid_pairs] = np.log(price_data[valid_pairs] / shifted_price[valid_pairs])
            
            # 무한값 제거
            log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
            
            # 충분한 데이터가 있는지 확인
            if len(log_returns.dropna()) < 60:
                print(f"{category} 카테고리는 유효한 로그 수익률 데이터가 부족합니다: {len(log_returns.dropna())} < 60")
                continue
            
            #print(f"{category} 카테고리의 유효한 로그 수익률 데이터 수: {len(log_returns.dropna())}")
           
            # 롤링 평균 및 표준편차 계산 (60일)
            rolling_mean = log_returns.rolling(window=60, min_periods=30).mean()
            rolling_std = log_returns.rolling(window=60, min_periods=30).std()
            
            # 표준편차가 0인 경우 건너뛰기
            rolling_std = rolling_std.replace(0, np.nan)
            
            # Z-score 계산 (60일 기준)
            z_scores = pd.Series(np.nan, index=log_returns.index)
            valid_indices = ~rolling_mean.isna() & ~rolling_std.isna() & ~log_returns.isna() & (rolling_std > 0)
            z_scores[valid_indices] = (log_returns[valid_indices] - rolling_mean[valid_indices]) / rolling_std[valid_indices]
            
            # NaN이나 무한대 값 처리
            z_scores = z_scores.replace([np.inf, -np.inf], np.nan).dropna()
            
            #print(f"{category} 카테고리의 유효한 Z-score 데이터 수: {len(z_scores)}")

            # 극단적 이벤트 감지 (Z-score의 절대값이 임계값 이상)
            extreme_events = z_scores[z_scores.abs() > std_threshold]
            #print(f"{category} 카테고리에서 감지된 극단적 이벤트 수: {len(extreme_events)}")
            
            for date, z_score in extreme_events.items():
                try:
                    event_type = "Extreme Surge" if z_score > 0 else "Extreme Drop"
                    
                    # 안전하게 return_value 가져오기 - 인덱스 유효성 체크 강화
                    if date in log_returns.index:
                        return_value = log_returns.loc[date]
                        # 유효한 값인지 확인
                        if pd.notna(return_value) and np.isfinite(return_value):
                            return_pct = (np.exp(return_value) - 1) * 100  # 백분율로 변환
                        else:
                            return_pct = np.nan
                    else:
                        return_value = np.nan
                        return_pct = np.nan
                        
                    # 설명 생성 - NaN 값 안전하게 처리
                    if pd.notna(return_pct) and np.isfinite(return_pct):
                        description = f'{category} {event_type} (Daily changes: {return_pct:.2f}%, Z-score: {z_score:.2f})'
                    else:
                        description = f'{category} {event_type} (Z-score: {z_score:.2f})'

                    # 이벤트 추가 - 명시적으로 float 변환 및 None 대신 np.nan 사용
                    event_dict = {
                        'date': date,
                        'category': category,
                        'event_type': event_type,
                        'z_score': float(z_score),
                        'return_pct': float(return_pct) if pd.notna(return_pct) and np.isfinite(return_pct) else np.nan,
                        'description': description
                    }
                    black_swan_events.append(event_dict)
                    print(f"Detected Black swan event: {description}")
                    
                except Exception as e:
                    print(f"{category} 이벤트 처리 중 오류: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"{category} 카테고리 분석 중 오류: {str(e)}")
            continue
    
    # 2. 경제 지표 블랙 스완 이벤트 감지
    if daily_indicators is not None and not daily_indicators.empty:
        important_indicators = [col for col in daily_indicators.columns if any(
            ind in col for ind in [
                'GDP', 'CPI', 'Fed_Funds_Rate', 'Dollar_Index',
                'fear_greed_value', 'sentiment_score_mean'
            ]
        )]
        
        print(f"the number of important indicators: {len(important_indicators)}")
        
        for indicator in important_indicators:
            try:
                print(f"{indicator} analyse...")
                
                # 지표의 일별 변화량
                indicator_data = daily_indicators[indicator].dropna()
                
                # 데이터 충분성 확인
                if len(indicator_data) < 60:
                    print(f"{indicator} 지표는 데이터가 부족합니다: {len(indicator_data)} < 60")
                    continue
                
                # 일별 변화량 계산
                indicator_change = indicator_data.diff()
                indicator_change = indicator_change.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(indicator_change) < 60:
                    print(f"{indicator} 지표의 변화량 데이터가 부족합니다: {len(indicator_change)} < 60")
                    continue
                
                #print(f"{indicator} 지표의 유효한 변화량 데이터 수: {len(indicator_change)}")
                
                # 롤링 평균 및 표준편차 계산 (60일)
                rolling_mean = indicator_change.rolling(window=60, min_periods=30).mean()
                rolling_std = indicator_change.rolling(window=60, min_periods=30).std()
                
                # 표준편차가 0인 경우 건너뛰기
                rolling_std = rolling_std.replace(0, np.nan)
                
                # Z-score 계산 (안전한 방식)
                z_scores = pd.Series(np.nan, index=indicator_change.index)
                valid_indices = ~rolling_mean.isna() & ~rolling_std.isna() & ~indicator_change.isna() & (rolling_std > 0)
                z_scores[valid_indices] = (indicator_change[valid_indices] - rolling_mean[valid_indices]) / rolling_std[valid_indices]
                
                # 무한값 및 NaN 처리
                z_scores = z_scores.replace([np.inf, -np.inf], np.nan).dropna()
                
                #print(f"{indicator} 지표의 유효한 Z-score 데이터 수: {len(z_scores)}")
                
                # 극단적 이벤트 감지
                extreme_events = z_scores[z_scores.abs() > std_threshold]
                #print(f"{indicator} 지표에서 감지된 극단적 이벤트 수: {len(extreme_events)}")
                
                for date, z_score in extreme_events.items():
                    try:
                        event_type = "Extreme Surge" if z_score > 0 else "Extreme Drop"
                        
                        # 지표명 추출
                        if '_norm' in indicator:
                            indicator_name = indicator.replace('_norm', '')
                        else:
                            indicator_name = indicator
                        
                        # 이벤트 추가 - 명시적으로 float 변환
                        event_dict = {
                            'date': date,
                            'category': 'Economic Indicator',
                            'event_type': f'{indicator_name} {event_type}',
                            'z_score': float(return_pct) if pd.notna(return_pct) and np.isfinite(return_pct) else np.nan,
                            'return_pct': float(return_pct) if pd.notna(return_pct) and np.isfinite(return_pct) else np.nan,  # 지표는 수익률이 없으므로 np.nan 사용
                            'description': f'{indicator_name}: {event_type} (Z-score: {z_score:.2f})'
                        }
                        black_swan_events.append(event_dict)
                        #print(f"블랙 스완 이벤트 감지 (지표): {event_dict['description']}")
                        
                    except Exception as e:
                        print(f"{indicator} 이벤트 처리 중 오류: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"{indicator} 지표 분석 중 오류: {str(e)}")
                continue
    
    # 데이터프레임으로 변환
    if black_swan_events:
        try:
            black_swan_df = pd.DataFrame(black_swan_events)
            print(f"총 {len(black_swan_df)}개의 블랙 스완 이벤트가 감지되었습니다.")
            
            # 결측치 직접 확인
            for col in black_swan_df.columns:
                print(f"{col} 컬럼의 결측치 수: {black_swan_df[col].isna().sum()}")
            
            # NaN 값 처리 - 특히 numeric 컬럼에 대해
            black_swan_df['z_score'] = black_swan_df['z_score'].apply(lambda x: float(x) if pd.notna(x) and np.isfinite(x) else np.nan)
            black_swan_df['return_pct'] = black_swan_df['return_pct'].apply(lambda x: float(x) if pd.notna(x) and np.isfinite(x) else np.nan)
            
            # 날짜 기준으로 정렬
            black_swan_df = black_swan_df.sort_values('date', ascending=False)
            return black_swan_df
        except Exception as e:
            print(f"블랙 스완 이벤트 데이터프레임 생성 중 오류: {str(e)}")
            return pd.DataFrame(columns=['date', 'category', 'event_type', 'z_score', 'return_pct', 'description'])
    else:
        print("감지된 블랙 스완 이벤트가 없습니다.")
        return pd.DataFrame(columns=['date', 'category', 'event_type', 'z_score', 'return_pct', 'description'])

def detect_correlations_breakdown(daily_avg_prices, daily_indicators, window=30, threshold=0.25):
    """
    상관관계 붕괴 이벤트를 감지합니다 (일반적으로 강한 상관관계가 갑자기 약해지거나 반전되는 경우).
    
    Parameters:
    -----------
    daily_avg_prices : DataFrame
        일별 평균 가격 데이터
    daily_indicators : DataFrame
        일별 지표 데이터
    window : int
        상관관계 계산을 위한 롤링 윈도우 크기 (기본값: 30 - 이전 60에서 축소)
    threshold : float
        상관관계 변화 임계값 (기본값: 0.25 - 이전 0.4에서 낮춤)
    
    Returns:
    --------
    correlation_events_df : DataFrame
        감지된 상관관계 붕괴 이벤트를 포함하는 데이터프레임
    """
    correlation_events = []
    
    # 데이터 유효성 먼저 검사 
    if daily_avg_prices is None or daily_avg_prices.empty:
        print("일별 평균 가격 데이터가 비어 있습니다.")
        return pd.DataFrame(columns=['date', 'pair', 'event_type', 'old_correlation', 
                                    'new_correlation', 'change', 'description'])
    
    # 지표 데이터가 없어도 카테고리 간 상관관계는 계산 가능
    indicators_available = False
    if daily_indicators is not None and not daily_indicators.empty:
        indicators_available = True
    else:
        print("일별 지표 데이터가 없거나 비어 있습니다. 카테고리 간 상관관계만 분석합니다.")
    
    # ✅ datetime 형식으로 강제 변환
    try:
        if not pd.api.types.is_datetime64_any_dtype(daily_avg_prices.index):
            daily_avg_prices.index = pd.to_datetime(daily_avg_prices.index, errors='coerce')
        if indicators_available and not pd.api.types.is_datetime64_any_dtype(daily_indicators.index):
            daily_indicators.index = pd.to_datetime(daily_indicators.index, errors='coerce')
    except Exception as e:
        print(f"날짜 변환 중 오류 발생: {str(e)}")
        return pd.DataFrame(columns=['date', 'pair', 'event_type', 'old_correlation', 
                                   'new_correlation', 'change', 'description'])
    
    # 데이터 유효 범위 확인
    if len(daily_avg_prices) < window:
        print(f"데이터가 충분하지 않습니다: {len(daily_avg_prices)} < {window}")
        return pd.DataFrame(columns=['date', 'pair', 'event_type', 'old_correlation', 
                                   'new_correlation', 'change', 'description'])
    
    print(f"상관관계 붕괴 분석 시작: 윈도우={window}, 임계값={threshold}")
    print(f"일별 평균 가격 데이터 형태: {daily_avg_prices.shape}, 기간: {daily_avg_prices.index.min()} ~ {daily_avg_prices.index.max()}")
    
    # 데이터 전처리: 모든 카테고리에 대한 기본 정보 출력
    for col in daily_avg_prices.columns:
        nan_count = daily_avg_prices[col].isna().sum()
        zero_count = (daily_avg_prices[col] == 0).sum()
        #print(f"카테고리 '{col}': 총 {len(daily_avg_prices[col])}개 데이터, NaN {nan_count}개, 0값 {zero_count}개")
    
    # 1. 카테고리-카테고리 상관관계 붕괴 감지
    categories = daily_avg_prices.columns.tolist()
    if len(categories) < 2:
        print("카테고리 수가 충분하지 않습니다. 최소 2개 이상 필요합니다.")
        # 빈 데이터프레임 반환
        return pd.DataFrame(columns=['date', 'pair', 'event_type', 'old_correlation', 
                                   'new_correlation', 'change', 'description'])
    
    #print(f"분석할 카테고리 수: {len(categories)}")
    
    for i in range(len(categories)):
        for j in range(i+1, len(categories)):
            cat1 = categories[i]
            cat2 = categories[j]
            
            try:
                #print(f"\n{cat1}-{cat2} 카테고리 쌍 분석 중...")
                
                # 카테고리의 데이터 추출 (NaN만 제거, 0은 유지)
                prices1 = daily_avg_prices[cat1].dropna()
                prices2 = daily_avg_prices[cat2].dropna()
                
                # 데이터가 충분한지 확인
                if len(prices1) < window or len(prices2) < window:
                    print(f"{cat1}-{cat2} 카테고리 쌍은 데이터가 부족합니다: {cat1}={len(prices1)}, {cat2}={len(prices2)}, 필요={window}")
                    continue
                
                # 공통 인덱스 찾기
                common_dates = prices1.index.intersection(prices2.index)
                
                # 공통 인덱스가 충분한지 확인
                if len(common_dates) < window:
                    print(f"{cat1}-{cat2} 카테고리 쌍은 공통 인덱스가 부족합니다: {len(common_dates)} < {window}")
                    continue
                
                #print(f"{cat1}-{cat2} 카테고리 쌍의 공통 인덱스 수: {len(common_dates)}")
                
                # 공통 날짜에 대한 데이터 준비
                prices1_common = prices1[common_dates]
                prices2_common = prices2[common_dates]
                
                # 정규화된 가격을 사용하여 상관관계 계산 (단순 변화율 대신)
                # 이렇게 하면 스케일 차이로 인한 상관관계 왜곡을 줄일 수 있음
                prices1_norm = (prices1_common - prices1_common.mean()) / prices1_common.std()
                prices2_norm = (prices2_common - prices2_common.mean()) / prices2_common.std()
                
                # 데이터프레임으로 결합
                price_data = pd.DataFrame({
                    cat1: prices1_norm,
                    cat2: prices2_norm
                })
                
                # NaN 행 제거
                price_data = price_data.dropna()
                
                # 데이터가 충분한지 최종 확인
                if len(price_data) < window:
                    print(f"{cat1}-{cat2} 카테고리 쌍은 최종 데이터가 부족합니다: {len(price_data)} < {window}")
                    continue
                
                # 각 시리즈의 표준편차가 0인지 확인 - 정규화했으므로 이 부분은 생략 가능하지만 안전을 위해 유지
                if price_data[cat1].std() < 1e-8 or price_data[cat2].std() < 1e-8:
                    print(f"{cat1}-{cat2} 카테고리 쌍 중 하나가 상수값입니다.")
                    continue
                
                #print(f"{cat1}-{cat2} 카테고리 쌍의 유효한 데이터 수: {len(price_data)}")
                
                # 롤링 상관계수 계산
                # min_periods를 더 작게 설정하여 초기 데이터도 활용
                min_periods = max(5, window // 3)  # 최소 5개 또는 윈도우의 1/3
                rolling_corr = price_data[cat1].rolling(window=window, min_periods=min_periods).corr(price_data[cat2])
                
                # NaN 및 무한값 제거
                rolling_corr = rolling_corr.replace([np.inf, -np.inf], np.nan)
                
                # 핵심 디버깅 정보: 롤링 상관계수 통계
                valid_corr = rolling_corr.dropna()
                if len(valid_corr) > 0:
                    print(f"유효한 롤링 상관계수: {len(valid_corr)}개, 평균: {valid_corr.mean():.4f}, 최소: {valid_corr.min():.4f}, 최대: {valid_corr.max():.4f}")
                else:
                    print("유효한 롤링 상관계수가 없습니다.")
                    continue
                
                # 롤링 상관계수의 변화량 계산
                corr_change = rolling_corr.diff().abs()  # 절대값으로 변화량 계산
                
                # NaN 및 무한값 제거
                corr_change = corr_change.replace([np.inf, -np.inf], np.nan).dropna()
                
                # 상관관계 변화량이 임계값을 초과하는 날짜 찾기
                breakdown_dates = corr_change[corr_change > threshold].index
                
                #print(f"{cat1}-{cat2} 카테고리 쌍에서 감지된 상관관계 변화 이벤트 수: {len(breakdown_dates)}")
                
                # 변화량 통계: 임계값 조정에 도움
                if len(corr_change) > 0:
                    print(f"상관관계 변화량 통계: 평균: {corr_change.mean():.4f}, 최대: {corr_change.max():.4f}, 95% 백분위: {corr_change.quantile(0.95):.4f}")
                
                # 이벤트 생성
                for date in breakdown_dates:
                    try:
                        # 현재 날짜와 이전 날짜가 rolling_corr에 있는지 확인
                        if date not in rolling_corr.index:
                            continue
                            
                        # 이전 날짜 가져오기
                        date_idx = rolling_corr.index.get_loc(date)
                        if date_idx == 0:  # 첫 번째 행이면 이전 날짜가 없음
                            continue
                            
                        prev_date = rolling_corr.index[date_idx - 1]
                        
                        # 이전 및 현재 상관계수 값 가져오기
                        if prev_date not in rolling_corr.index or pd.isna(rolling_corr.loc[prev_date]):
                            continue
                            
                        old_corr = rolling_corr.loc[prev_date]
                        new_corr = rolling_corr.loc[date]
                        change = abs(new_corr - old_corr)  # 절대값 변화량
                        
                        # 값이 유효한지 확인
                        if not np.isfinite(old_corr) or not np.isfinite(new_corr) or not np.isfinite(change):
                            continue
                        
                        # 변화 방향 결정
                        if new_corr < old_corr:
                            direction = "감소"
                            event_type = "Correlation Breakdown"
                        else:
                            direction = "증가"
                            event_type = "Correlation Spike"
                        
                        # 이벤트 정보 저장
                        correlation_events.append({
                            'date': date,
                            'pair': f'{cat1}-{cat2}',
                            'event_type': event_type,
                            'old_correlation': float(old_corr),
                            'new_correlation': float(new_corr),
                            'change': float(change),
                            'description': f'{cat1}와 {cat2} 간의 상관관계 {direction} ({old_corr:.2f} → {new_corr:.2f}, 변화량: {change:.2f})'
                        })
                        
                        print(f"상관관계 변화 이벤트 감지: {cat1}와 {cat2} 간의 상관관계 {direction} ({old_corr:.2f} → {new_corr:.2f}, 변화량: {change:.2f})")
                    except Exception as e:
                        print(f"{cat1}-{cat2} 상관관계 이벤트 처리 중 오류: {str(e)}")
                        continue
            except Exception as e:
                print(f"{cat1}-{cat2} 상관관계 분석 중 오류: {str(e)}")
                continue
    
    # 2. 카테고리-지표 상관관계 붕괴 감지 (지표 데이터가 있는 경우에만)
    if indicators_available:
        # 중요 지표 필터링
        important_indicators = [col for col in daily_indicators.columns if any(
            ind in col for ind in [
                'GDP', 'CPI', 'Fed_Funds_Rate', 'Dollar_Index',
                'fear_greed_value', 'sentiment_score_mean'
            ]
        )]
        
        #print(f"\n분석할 중요 지표 수: {len(important_indicators)}")
        
        for category in categories:
            for indicator in important_indicators:
                try:
                    print(f"\n{category}-{indicator} 카테고리-지표 쌍 분석 중...")
                    
                    # 카테고리와 지표 데이터 추출 (NaN만 제거)
                    prices = daily_avg_prices[category].dropna()
                    indicator_values = daily_indicators[indicator].dropna()
                    
                    # 데이터가 충분한지 확인
                    if len(prices) < window or len(indicator_values) < window:
                        print(f"{category}-{indicator} 카테고리-지표 쌍은 데이터가 부족합니다.")
                        continue
                    
                    # 공통 인덱스 찾기
                    common_dates = prices.index.intersection(indicator_values.index)
                    
                    # 공통 인덱스가 충분한지 확인
                    if len(common_dates) < window:
                        print(f"{category}-{indicator} 카테고리-지표 쌍은 공통 인덱스가 부족합니다: {len(common_dates)} < {window}")
                        continue
                    
                    #print(f"{category}-{indicator} 카테고리-지표 쌍의 공통 인덱스 수: {len(common_dates)}")
                    
                    # 공통 날짜에 대한 데이터 준비
                    prices_common = prices[common_dates]
                    indicator_common = indicator_values[common_dates]
                    
                    # 정규화된 데이터 사용
                    prices_norm = (prices_common - prices_common.mean()) / prices_common.std()
                    indicator_norm = (indicator_common - indicator_common.mean()) / indicator_common.std()
                    
                    # 데이터프레임으로 결합
                    combined_data = pd.DataFrame({
                        'prices': prices_norm,
                        'indicator': indicator_norm
                    })
                    
                    # NaN 행 제거
                    combined_data = combined_data.dropna()
                    
                    # 데이터가 충분한지 최종 확인
                    if len(combined_data) < window:
                        print(f"{category}-{indicator} 카테고리-지표 쌍은 최종 데이터가 부족합니다: {len(combined_data)} < {window}")
                        continue
                    
                    # 각 시리즈의 표준편차가 0인지 확인
                    if combined_data['prices'].std() < 1e-8 or combined_data['indicator'].std() < 1e-8:
                        print(f"{category}-{indicator} 카테고리-지표 쌍 중 하나가 상수값입니다.")
                        continue
                    
                    #print(f"Between {category}-{indicator} combined & available data : {len(combined_data)}")
                    
                    # 롤링 상관계수 계산
                    min_periods = max(5, window // 3)
                    rolling_corr = combined_data['prices'].rolling(window=window, min_periods=min_periods).corr(combined_data['indicator'])
                    
                    # NaN 및 무한값 제거
                    rolling_corr = rolling_corr.replace([np.inf, -np.inf], np.nan)
                    
                    # # 핵심 디버깅 정보: 롤링 상관계수 통계
                    # valid_corr = rolling_corr.dropna()
                    # if len(valid_corr) > 0:
                    #     print(f"유효한 롤링 상관계수: {len(valid_corr)}개, 평균: {valid_corr.mean():.4f}, 최소: {valid_corr.min():.4f}, 최대: {valid_corr.max():.4f}")
                    # else:
                    #     print("유효한 롤링 상관계수가 없습니다.")
                    #     continue
                    
                    # 롤링 상관계수의 변화량 계산
                    corr_change = rolling_corr.diff().abs()  # 절대값으로 변화량 계산
                    
                    # NaN 및 무한값 제거
                    corr_change = corr_change.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    # 상관관계 변화량이 임계값을 초과하는 날짜 찾기
                    breakdown_dates = corr_change[corr_change > threshold].index
                    
                    #print(f"{category}-{indicator} 카테고리-지표 쌍에서 감지된 상관관계 변화 이벤트 수: {len(breakdown_dates)}")
                    
                    # 변화량 통계
                    if len(corr_change) > 0:
                        print(f"상관관계 변화량 통계: 평균: {corr_change.mean():.4f}, 최대: {corr_change.max():.4f}, 95% 백분위: {corr_change.quantile(0.95):.4f}")
                    
                    # 이벤트 생성
                    for date in breakdown_dates:
                        try:
                            # 현재 날짜와 이전 날짜가 rolling_corr에 있는지 확인
                            if date not in rolling_corr.index:
                                continue
                                
                            # 이전 날짜 가져오기
                            date_idx = rolling_corr.index.get_loc(date)
                            if date_idx == 0:  # 첫 번째 행이면 이전 날짜가 없음
                                continue
                                
                            prev_date = rolling_corr.index[date_idx - 1]
                            
                            # 이전 및 현재 상관계수 값 가져오기
                            if prev_date not in rolling_corr.index or pd.isna(rolling_corr.loc[prev_date]):
                                continue
                                
                            old_corr = rolling_corr.loc[prev_date]
                            new_corr = rolling_corr.loc[date]
                            change = abs(new_corr - old_corr)  # 절대값 변화량
                            
                            # 값이 유효한지 확인
                            if not np.isfinite(old_corr) or not np.isfinite(new_corr) or not np.isfinite(change):
                                continue
                            
                            # 변화 방향 결정
                            if new_corr < old_corr:
                                direction = "감소"
                                event_type = "Correlation Breakdown"
                            else:
                                direction = "증가"
                                event_type = "Correlation Spike"
                            
                            # 이벤트 정보 저장
                            correlation_events.append({
                                'date': date,
                                'pair': f'{category}-{indicator}',
                                'event_type': event_type,
                                'old_correlation': float(old_corr),
                                'new_correlation': float(new_corr),
                                'change': float(change),
                                'description': f'Correlation between {category} and {indicator} : {direction} ({old_corr:.2f} → {new_corr:.2f}, Change: {change:.2f})'
                            })
                            
                            #print(f"상관관계 변화 이벤트 감지: {category}와 {indicator} 간의 상관관계 {direction} ({old_corr:.2f} → {new_corr:.2f}, 변화량: {change:.2f})")
                        except Exception as e:
                            #print(f"{category}-{indicator} 상관관계 이벤트 처리 중 오류: {str(e)}")
                            continue
                except Exception as e:
                    print(f"{category}-{indicator} 상관관계 분석 중 오류: {str(e)}")
                    continue
   
        # detect_correlations_breakdown 함수 마지막 부분 수정
    if correlation_events:
        try:
            correlation_breakdown_df = pd.DataFrame(correlation_events)  # 변수명 변경
            #print(f"\n총 {len(correlation_breakdown_df)}개의 상관관계 변화 이벤트가 감지되었습니다.")
            
            # 결측치 직접 확인
            for col in correlation_breakdown_df.columns:
                print(f"{col} 컬럼의 결측치 수: {correlation_breakdown_df[col].isna().sum()}")
            
            # NaN 값 처리 - 특히 numeric 컬럼에 대해
            numeric_cols = ['old_correlation', 'new_correlation', 'change']
            for col in numeric_cols:
                correlation_breakdown_df[col] = correlation_breakdown_df[col].apply(
                    lambda x: float(x) if pd.notna(x) and np.isfinite(x) else np.nan
                )
            
            # 날짜 기준으로 정렬
            correlation_breakdown_df = correlation_breakdown_df.sort_values('date', ascending=False)
            return correlation_breakdown_df    
        except Exception as e:
            print(f"결과 데이터프레임 생성 중 오류: {str(e)}")
            return pd.DataFrame(columns=['date', 'pair', 'event_type', 'old_correlation', 
                                    'new_correlation', 'change', 'description'])
    else:
        print("\n감지된 상관관계 변화 이벤트가 없습니다.")
        return pd.DataFrame(columns=['date', 'pair', 'event_type', 'old_correlation', 
                                    'new_correlation', 'change', 'description'])

def generate_extreme_events_features(df, extreme_events_df, black_swan_df, correlation_breakdown_df=None):
    """
    극단적 이벤트 특성을 생성합니다 (모델 훈련에 사용).
    
    Parameters:
    -----------
    df : DataFrame
        원본 데이터프레임
    extreme_events_df : DataFrame
        감지된 극단적 이벤트 데이터프레임
    black_swan_df : DataFrame
        감지된 블랙 스완 이벤트 데이터프레임
    correlation_breakdown_df : DataFrame, optional
        감지된 상관관계 붕괴 이벤트 데이터프레임
    
    Returns:
    --------
    df : DataFrame
        극단적 이벤트 특성이 추가된 데이터프레임
    """
    if df is None or df.empty:
        return df
    
    # 날짜 컬럼 datetime으로 강제 변환
    try:
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if black_swan_df is not None and 'date' in black_swan_df.columns and not pd.api.types.is_datetime64_any_dtype(black_swan_df['date']):
            black_swan_df['date'] = pd.to_datetime(black_swan_df['date'], errors='coerce')
        if extreme_events_df is not None and 'date' in extreme_events_df.columns and not pd.api.types.is_datetime64_any_dtype(extreme_events_df['date']):
            extreme_events_df['date'] = pd.to_datetime(extreme_events_df['date'], errors='coerce')
        if correlation_breakdown_df is not None and 'date' in correlation_breakdown_df.columns and not pd.api.types.is_datetime64_any_dtype(correlation_breakdown_df['date']):
            correlation_breakdown_df['date'] = pd.to_datetime(correlation_breakdown_df['date'], errors='coerce')
    except Exception as e:
        st.error(f"An error occurred during date conversion: {str(e)}")
        return df

    # 1. 극단적 이벤트 특성
    try:
        if extreme_events_df is not None and not extreme_events_df.empty:
            # 경제 지표 극단값 표시
            economic_extreme_events = extreme_events_df[
                (extreme_events_df['event_type'] == 'Extreme High') | 
                (extreme_events_df['event_type'] == 'Extreme Low')
            ]
            
            if not economic_extreme_events.empty:
                # 극단 이벤트 발생 여부를 나타내는 특성 생성
                extreme_dates = economic_extreme_events['date'].unique()
                
                # 데이터프레임에 극단값 표시자 추가
                df['extreme_economic_event'] = df['date'].isin(extreme_dates).astype(int)
                
                # 연속된 극단 이벤트 일수 계산
                df['extreme_event_days'] = 0
                current_count = 0
                
                for i, row in df.sort_values('date').iterrows():
                    if row['extreme_economic_event'] == 1:
                        current_count += 1
                    else:
                        current_count = 0
                    df.at[i, 'extreme_event_days'] = current_count
    except Exception as e:
        st.warning(f"Error processing extreme events: {str(e)}")

    # 2. 블랙 스완 이벤트 특성
    try:
        if black_swan_df is not None and not black_swan_df.empty:
            # 블랙 스완 발생 여부를 나타내는 특성 생성
            black_swan_dates = black_swan_df['date'].unique()
            
            # 데이터프레임에 블랙 스완 표시자 추가
            df['black_swan_event'] = df['date'].isin(black_swan_dates).astype(int)
            
            # 블랙 스완 후 경과 일수 계산
            df = df.sort_values('date').reset_index(drop=True)
            df['days_since_black_swan'] = 999  # 기본값 설정 default
            last_black_swan_date = None
            
            for date in df['date'].unique():
                if date in black_swan_dates:
                    last_black_swan_date = date
                    df.loc[df['date'] == date, 'days_since_black_swan'] = 0
                elif last_black_swan_date is not None:
                    days_diff = (date - last_black_swan_date).days
                    df.loc[df['date'] == date, 'days_since_black_swan'] = days_diff
    except Exception as e:
        st.warning(f"Error processing black swan events: {str(e)}")
    # 3. 각 카테고리별 최근 극단적 이벤트 거리 계산
    try:
        if 'category' in df.columns and black_swan_df is not None and not black_swan_df.empty:
            categories = df['category'].unique()

            df['days_since_category_balck_swan'] = 999
            
            for category in categories:
                # 카테고리별 블랙 스완 이벤트
                category_black_swans = black_swan_df[black_swan_df['category'] == category]

                if not category_black_swans.empty:
                        # 각 날짜별로 가장 가까운 블랙 스완 이벤트와의 거리 계산
                        df_category = df[df['category'] == category].sort_values('date')
                        
                        for idx in df_category.index:
                            current_date = df_category.loc[idx, 'date']
                            
                            # 이 날짜 이전의 가장 최근 블랙 스완 찾기
                            prev_black_swans = category_black_swans[category_black_swans['date'] <= current_date]
                            
                            if not prev_black_swans.empty:
                                latest_black_swan = prev_black_swans['date'].max()
                                days_diff = (current_date - latest_black_swan).days
                                df.at[i, 'days_since_category_black_swan'] = days_diff
    except Exception as e:
        st.warning(f"Error processing category-specific black swan events: {str(e)}")

    # 4. 상관관계 붕괴 이벤트 특성
    try:
        if correlation_breakdown_df is not None and not correlation_breakdown_df.empty:
            # 상관관계 붕괴 발생 여부를 나타내는 특성 생성
            breakdown_dates = correlation_breakdown_df['date'].unique()
            
            # 데이터프레임에 상관관계 붕괴 표시자 추가
            df['correlation_breakdown'] = df['date'].isin(breakdown_dates).astype(int)
    except Exception as e:
        st.warning(f"Error processing correlation breakdown events: {str(e)}")
    
    # 최종적으로 채워지지 않은 값을 0으로 설정
    for col in ['extreme_economic_event', 'black_swan_event', 'correlation_breakdown']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 결측치가 있는 다른 열들은 적절한 값으로 채움
    for col in ['extreme_event_days', 'days_since_black_swan', 'days_since_category_black_swan']:
        if col in df.columns:
            df[col] = df[col].fillna(999)  # 높은 값으로 설정 (오래 전에 발생)
    
    return df
def visualize_extreme_events(extreme_events_df, black_swan_df, daily_avg_prices, daily_indicators, technical_patterns_df=None, category=None):
    """
    극단적 이벤트와 블랙 스완 이벤트를 시각화합니다.
    
    Parameters:
    -----------
    extreme_events_df : DataFrame
        감지된 극단적 이벤트 데이터프레임
    black_swan_df : DataFrame
        감지된 블랙 스완 이벤트 데이터프레임
    daily_avg_prices : DataFrame
        일별 평균 가격 데이터
    daily_indicators : DataFrame
        일별 지표 데이터
    technical_patterns_df : DataFrame, optional
        감지된 기술적 패턴 데이터프레임
    category : str, optional
        특정 카테고리에 대한 시각화 (기본값: None)
    """
    st.subheader("Extreme Event and Visualization")

    try:
        if daily_avg_prices is None or daily_indicators is None:
            st.warning("No Price data or Indicator data")
            return
        
        if extreme_events_df is not None and 'date' in extreme_events_df.columns and not pd.api.types.is_datetime64_any_dtype(extreme_events_df['date']):
            extreme_events_df['date'] = pd.to_datetime(extreme_events_df['date'], errors='coerce')

        if black_swan_df is not None and 'date' in black_swan_df.columns and not pd.api.types.is_datetime64_any_dtype(black_swan_df['date']):
            black_swan_df['date'] = pd.to_datetime(black_swan_df['date'], errors='coerce')

        if daily_avg_prices is not None and not pd.api.types.is_datetime64_any_dtype(daily_avg_prices.index):
            daily_avg_prices.index = pd.to_datetime(daily_avg_prices.index, errors='coerce')

        if daily_indicators is not None and not pd.api.types.is_datetime64_any_dtype(daily_indicators.index):
            daily_indicators.index = pd.to_datetime(daily_indicators.index, errors='coerce')

        # 필터링
        if category:
            if black_swan_df is not None and not black_swan_df.empty:
                black_swan_df = black_swan_df[black_swan_df['category'] == category]

    except Exception as e:
        st.error(f"An error occurred during data preprocessing: {str(e)}")
        return
    # 1. Price Trends Around Black Swan Events
    # 1. Price Trends Around Black Swan Events
    if daily_avg_prices is not None and not daily_avg_prices.empty:
        st.write("### Price Trends Around Black Swan Events")
        
        categories_to_plot = [category] if category else daily_avg_prices.columns[:min(4, len(daily_avg_prices.columns))]
        
        # 격자 형식으로 표시하기 위한 준비
        num_categories = len(categories_to_plot)
        cols_per_row = 2  # 한 줄에 2개의 그래프
        num_rows = (num_categories + cols_per_row - 1) // cols_per_row  # 올림 나눗셈
        
        # 각 행마다 2개의 컬럼 생성
        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            
            for col_idx in range(cols_per_row):
                cat_idx = row * cols_per_row + col_idx
                
                # 카테고리 인덱스가 유효한지 확인
                if cat_idx < num_categories:
                    cat = categories_to_plot[cat_idx]
                    
                    with cols[col_idx]:
                        try:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            # 가격 데이터 플롯
                            ax.plot(daily_avg_prices.index, daily_avg_prices[cat], label=cat, color='blue')
                            
                            # 블랙 스완 이벤트 표시
                            if black_swan_df is not None and not black_swan_df.empty:
                                cat_black_swans = black_swan_df[black_swan_df['category'] == cat]
                                
                                for _, event in cat_black_swans.iterrows():
                                    event_date = event['date']
                                    if event_date in daily_avg_prices.index:
                                        price_at_event = daily_avg_prices.loc[event_date, cat]
                                        event_type = event['event_type']
                                        color = 'red' if 'Down' in event_type else 'green'
                                        
                                        ax.scatter(event_date, price_at_event, color=color, s=100, zorder=5)
                                        ax.annotate(f"{event_type} ({event['z_score']:.1f})", 
                                                (event_date, price_at_event),
                                                textcoords="offset points", 
                                                xytext=(0, 10), 
                                                ha='center')
                            
                            # 20일 이동평균선 추가
                            ma_20 = daily_avg_prices[cat].rolling(window=20).mean()
                            ax.plot(daily_avg_prices.index, ma_20, label='20-day moving average', color='orange', linestyle='--')
                            
                            # 50일 이동평균선 추가
                            ma_50 = daily_avg_prices[cat].rolling(window=50).mean()
                            ax.plot(daily_avg_prices.index, ma_50, label='50-day moving average', color='green', linestyle='--')
                            
                            ax.set_title(f'{cat} Price trend & black swan event', fontsize=16)
                            ax.set_xlabel('Date', fontsize=14)
                            ax.set_ylabel('Price', fontsize=14)
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            
                            # x축 날짜 형식 개선
                            plt.xticks(rotation=45)
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                        except Exception as e:
                            st.error(f"{cat} 시각화 중 오류 발생: {str(e)}")
    
    st.markdown("""
    # 📊 Analyse von Black-Swan-Ereignissen nach Anlageklassen und globalen Themen

    ---

    ## 🛡️ Verteidigungssektor (Defense)

    | 📅 Datum | 📈 Ereignistyp | 📊 Z-Score | 🌎 Globale Themen |
    |:--------|:--------------|:----------|:-----------------|
    | 2023-04 | Extremer Rückgang | -3,1 | Stillstand im Russland-Ukraine-Krieg, Unsicherheit über westliche Unterstützung |
    | 2023-10 | Extremer Anstieg | 5,9 | Ausbruch des Israel-Hamas-Krieges, Aussicht auf erhöhte globale Verteidigungsausgaben |
    | 2024-01 | Extremer Rückgang | -3,6 | Nachlassende Besorgnis über Nahost-Konflikt, Ukraine-Unterstützungsmüdigkeit |
    | 2024-07 | Extremer Anstieg | 3,2 | NATO-Erweiterung, Verlängerung des Ukraine-Krieges |
    | 2024-11 | Extremer Anstieg | 4,3 | Verstärkte Verteidigungsbudgets nach US-Wahl |
    | 2025-01 | Extremer Anstieg | 4,1 | Verteidigungsorientierte Politik der neuen Regierung |

    **🔎 Anlageimplikationen:**  
    - Strukturelles Aufwärtsmomentum bei geopolitischen Krisen  
    - Aufwärtsereignisse stärker ausgeprägt als Rückgänge

    ---

    ## 🪙 Goldmarkt (Gold)

    | 📅 Datum | 📈 Ereignistyp | 📊 Z-Score | 🌎 Globale Themen |
    |:--------|:--------------|:----------|:-----------------|
    | 2023-09 | Extremer Anstieg | 3,2 | Höhepunkt der Zinserhöhungen, Fed-Lockerungserwartungen, Nahost-Spannungen |
    | 2024-05 | Extremer Rückgang | -3,6 | Steigende Inflation, starke US-Dollar-Performance |
    | 2024-11 | Extremer Rückgang | -3,6 | US-Wahlergebnis, Unsicherheitsrückgang, Risikoanlagen bevorzugt |

    **🔎 Anlageimplikationen:**  
    - Hohe Sensitivität gegenüber geopolitischen und geldpolitischen Faktoren  
    - Langfristiger Aufwärtstrend trotz kurzfristiger Schwankungen

    ---

    ## 🤖 KI-Technologieaktien (Tech_AI)

    | 📅 Datum | 📈 Ereignistyp | 📊 Z-Score | 🌎 Globale Themen |
    |:--------|:--------------|:----------|:-----------------|
    | 2023-06 | Extremer Rückgang | -3,8 | Abkühlung der KI-Euphorie, anhaltende Fed-Straffung |
    | 2023-12 | Extremer Rückgang | -3,0 | Gewinnmitnahmen zum Jahresende, schwache KI-Gewinne |
    | 2024-03 | Extremer Anstieg | 3,3 | Starke KI-Firmenzahlen, Zinssenkungserwartungen |
    | 2024-06 | Extremer Rückgang | -4,0 | Regulierungsängste im KI-Sektor, Inflation steigt |
    | 2024-10 | Extremer Anstieg | 3,5 | Tech-Rally vor US-Wahl |
    | 2025-02 | Extremer Rückgang | -3,4 | Regulierungsängste, Überbewertungsbedenken |

    **🔎 Anlageimplikationen:**  
    - Höchste Volatilität, höchste Ertragschancen  
    - Extrem abhängig von Zinspolitik, Innovation und Regulierung

    ---

    ## 📈 Vergleich der Anlageklassen & Investmentstrategie

    | 🏦 Anlageklasse | 📈 Anstieg (2023–2025) | 🕊️ Black-Swan-Charakteristik | 🎯 Sensitivitätsfaktoren |
    |:---------------|:----------------------|:----------------------------|:------------------------|
    | Verteidigungssektor | ca. 93% | Überwiegend Aufwärtsereignisse | Geopolitische Spannungen, Verteidigungsbudgets |
    | Gold | ca. 67% | Gemischte Auf- und Abwärtsereignisse | Inflation, Geldpolitik, Krisen |
    | KI-Technologieaktien | ca. 200% | Höchste Volatilität | Zinspolitik, Innovation, Regulierung |

    **🚀 Cross-Asset-Investmentstrategie:**  
    - 🔺 Geopolitische Spannungen/Wirtschaftliche Unsicherheit ↑ → Defense + Gold aufstocken  
    - 📈 Innovationsgetriebene Boom-Phase erwartet → Tech-Anteil erhöhen  
    - 🛡️ Black-Swan-Risiken ➔ Durch **breite Diversifikation** abfedern

    ---
    """)

        
    # 2. 극단적 경제 지표 이벤트 시각화
    if extreme_events_df is not None and not extreme_events_df.empty and daily_indicators is not None:
        economic_extremes = extreme_events_df[
            (extreme_events_df['event_type'] == 'Extreme High') | 
            (extreme_events_df['event_type'] == 'Extreme Low')
        ]
        
        if not economic_extremes.empty:
            st.write("### Extreme Event & Economic Indicator")
            
            # 상위 빈도 지표 선택
            top_indicators = economic_extremes['indicator'].value_counts().head(4).index.tolist()
            
            # 격자 형식으로 표시하기 위한 준비
            num_indicators = len(top_indicators)
            cols_per_row = 2  # 한 줄에 2개의 그래프
            num_rows = (num_indicators + cols_per_row - 1) // cols_per_row  # 올림 나눗셈
            
            # 각 행마다 2개의 컬럼 생성
            for row in range(num_rows):
                cols = st.columns(cols_per_row)
                
                for col_idx in range(cols_per_row):
                    ind_idx = row * cols_per_row + col_idx
                    
                    # 지표 인덱스가 유효한지 확인
                    if ind_idx < num_indicators:
                        indicator = top_indicators[ind_idx]
                        
                        with cols[col_idx]:
                            try:
                                # 지표명에서 '_norm'를 제거
                                clean_indicator = indicator.replace('_norm', '')
                                
                                # 원본 지표 데이터 추출
                                indicator_data = daily_indicators[indicator] if indicator in daily_indicators.columns else None
                                
                                if indicator_data is None:
                                    st.warning(f"{indicator} 데이터를 찾을 수 없습니다.")
                                    continue
                                
                                fig, ax = plt.subplots(figsize=(8, 6))
                                
                                # 지표 데이터 플롯
                                ax.plot(daily_indicators.index, indicator_data, label=clean_indicator, color='blue')
                                
                                # 극단적 이벤트 표시
                                indicator_extremes = economic_extremes[economic_extremes['indicator'] == indicator]
                                
                                for _, event in indicator_extremes.iterrows():
                                    event_date = event['date']
                                    if event_date in daily_indicators.index:
                                        value_at_event = indicator_data.loc[event_date]
                                        event_type = event['event_type']
                                        color = 'red' if event_type == 'Extreme High' else 'green'
                                        
                                        ax.scatter(event_date, value_at_event, color=color, s=60, zorder=2.5)
                                
                                # 90% 및 10% 분위수 라인 추가
                                high_threshold = indicator_data.quantile(0.90)
                                low_threshold = indicator_data.quantile(0.10)
                                
                                ax.axhline(y=high_threshold, color='red', linestyle='--', 
                                        label=f'top 10% threshold ({high_threshold:.2f})')
                                ax.axhline(y=low_threshold, color='green', linestyle='--', 
                                        label=f'low 10% threshold ({low_threshold:.2f})')
                                
                                ax.set_title(f'{clean_indicator} trend & extreme event', fontsize=16)
                                ax.set_xlabel('date', fontsize=14)
                                ax.set_ylabel(clean_indicator, fontsize=14)
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                
                                # x축 날짜 형식 개선
                                plt.xticks(rotation=45)
                                fig.tight_layout()
                                
                                st.pyplot(fig)
                                plt.close()
                            except Exception as e:
                                st.error(f"{indicator} 시각화 중 오류 발생: {str(e)}")

    # 3. 다차원 분석: 블랙 스완 이벤트와 지표 상관관계
    if black_swan_df is not None and not black_swan_df.empty:
        st.write("### 📊 Indicator Trends Before & After Black Swan Events")

        try:
            # 인덱스가 datetime 형식인지 확인하고 변환
            if not pd.api.types.is_datetime64_any_dtype(daily_indicators.index):
                daily_indicators.index = pd.to_datetime(daily_indicators.index, errors='coerce')

            # 블랙 스완 이벤트 날짜 변환
            black_swan_df['date'] = pd.to_datetime(black_swan_df['date'], errors='coerce')
            bs_dates = black_swan_df['date'].dropna().tolist()
            bs_types = black_swan_df['event_type'].tolist() if len(black_swan_df) > 0 else []
            
            # 블랙 스완 이벤트 카테고리 추가
            bs_categories = black_swan_df['category'].tolist() if 'category' in black_swan_df.columns and len(black_swan_df) > 0 else []

            # 분석 범위
            window_before = 20
            window_after = 20

            if daily_indicators is not None and len(bs_dates) > 0:
                # 주요 지표 선택
                key_indicators = [col for col in daily_indicators.columns if any(
                    ind in col for ind in [
                        'GDP_norm', 'CPI_norm', 'Dollar_Index_norm',
                        'fear_greed_value_norm', 'sentiment_score_mean_norm'
                    ]
                )][:4]

                if key_indicators:
                    # 격자 형식으로 표시하기 위한 준비
                    num_indicators = len(key_indicators)
                    cols_per_row = 2  # 한 줄에 2개의 그래프
                    num_rows = (num_indicators + cols_per_row - 1) // cols_per_row  # 올림 나눗셈
                    
                    # 각 행마다 2개의 컬럼 생성
                    for row in range(num_rows):
                        cols = st.columns(cols_per_row)
                        
                        for col_idx in range(cols_per_row):
                            ind_idx = row * cols_per_row + col_idx
                            
                            # 지표 인덱스가 유효한지 확인
                            if ind_idx < num_indicators:
                                indicator = key_indicators[ind_idx]
                                ind_name = indicator.split('_')[0]  # 지표 이름
                                
                                with cols[col_idx]:
                                    try:
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        
                                        # 명확한 색상 설정
                                        event_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
                                        
                                        # 각 이벤트별 데이터 시각화
                                        max_events = min(len(bs_dates), 3)  # 최대 3개 이벤트만 표시
                                        
                                        for j in range(max_events):
                                            if j < len(bs_dates) and j < len(bs_types):
                                                event_date = bs_dates[j]
                                                event_type = bs_types[j]
                                                event_category = bs_categories[j] if j < len(bs_categories) else "Unknown"
                                                
                                                # 유효한 datetime인지 확인
                                                if pd.isnull(event_date) or not isinstance(event_date, pd.Timestamp):
                                                    continue

                                                start_date = event_date - pd.Timedelta(days=window_before)
                                                end_date = event_date + pd.Timedelta(days=window_after)

                                                # 날짜 범위의 지표 데이터 추출
                                                mask = (daily_indicators.index >= start_date) & (daily_indicators.index <= end_date)
                                                period_data = daily_indicators.loc[mask, indicator]

                                                if not period_data.empty:
                                                    days_from_event = [(d - event_date).days for d in period_data.index]
                                                    
                                                    # 명확한 색상 할당 및 더 구체적인 레이블 사용
                                                    color = event_colors[j % len(event_colors)]
                                                    event_label = f'Event {j+1}: {event_type} ({event_category})'
                                                    
                                                    ax.plot(days_from_event, period_data.values,
                                                        label=event_label, color=color, linewidth=2)

                                                    # 이벤트 발생일 값 표시
                                                    if event_date in period_data.index:
                                                        event_idx = period_data.index.get_loc(event_date)
                                                        value_at_event = period_data.iloc[event_idx]
                                                        
                                                        ax.scatter(0, value_at_event, 
                                                                color=color, s=100, zorder=5)

                                        # Y축 자동 조정 (데이터 범위에 맞게)
                                        if len(ax.get_lines()) > 0:  # 라인이 있는지 확인
                                            y_data = [line.get_ydata() for line in ax.get_lines()]
                                            if y_data and all(len(data) > 0 for data in y_data):
                                                # 모든 라인의 y값 범위 계산
                                                all_y = np.concatenate(y_data)
                                                if len(all_y) > 0:
                                                    # Y축 범위 설정 (약간의 여백을 둠)
                                                    y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
                                                    y_range = y_max - y_min
                                                    if y_range > 0.0001:  # 변동이 있는 경우에만 조정
                                                        ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.1)
                                        
                                        ax.set_title(f'{ind_name} Before & After Black Swan Events', fontsize=14)
                                        ax.set_xlabel('Days from Event (0 = Event)', fontsize=12)
                                        ax.set_ylabel(ind_name, fontsize=12)
                                        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                                        ax.legend()
                                        ax.grid(True, alpha=0.3)
                                        
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close()
                                    except Exception as e:
                                        st.error(f"❌ {ind_name} 시각화 중 오류 발생: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error while analyzing before/after Black Swan events: {str(e)}")

    # 4. 기술적 지표 패턴과 극단적 이벤트의 관계
    if isinstance(technical_patterns_df, pd.DataFrame) and not technical_patterns_df.empty:
        if category:
            technical_patterns_df = technical_patterns_df[technical_patterns_df['category'] == category]
        
        if not technical_patterns_df.empty:
            st.write("### How Technical Patterns Relate to Extreme Market Events")
            
            try:
                # 시각화를 두 컬럼으로 나누기
                col1, col2 = st.columns(2)
                
                with col1:
                    # 패턴 유형별 빈도수
                    pattern_counts = technical_patterns_df['pattern'].value_counts()
                    
                    # 막대 그래프로 시각화
                    fig, ax = plt.subplots(figsize=(8, 6))
                    pattern_counts.plot(kind='bar', color='teal', ax=ax)
                    
                    plt.title('Frequency of technical pattern occurrence', fontsize=16)
                    plt.xlabel('Pattern Types', fontsize=14)
                    plt.ylabel('Frequency of occurrence', fontsize=14)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    plt.close()

                # 기술적 패턴 발생 날짜와 블랙 스완 이벤트의 관계 분석
                if isinstance(black_swan_df, pd.DataFrame) and not black_swan_df.empty:
                    try:
                        with col2:
                            # 날짜 컬럼을 안전하게 datetime으로 변환
                            technical_patterns_df['date'] = pd.to_datetime(technical_patterns_df['date'], errors='coerce')
                            black_swan_df['date'] = pd.to_datetime(black_swan_df['date'], errors='coerce')

                            # 유효한 날짜만 필터링
                            pattern_dates = technical_patterns_df['date'].dropna().unique()
                            black_swan_dates = black_swan_df['date'].dropna().unique()

                            # 분석에 사용할 일 수 범위
                            days_thresholds = [1, 3, 5, 10, 20]
                            pattern_types = technical_patterns_df['pattern'].unique()
                            black_swan_counts = {}

                            for pattern in pattern_types:
                                black_swan_counts[pattern] = []
                                # 해당 패턴에 해당하는 날짜만 추출하고 datetime으로 변환
                                pattern_specific_dates = pd.to_datetime(
                                    technical_patterns_df[technical_patterns_df['pattern'] == pattern]['date'],
                                    errors='coerce'
                                ).dropna()

                                for days in days_thresholds:
                                    count = 0
                                    for date in pattern_specific_dates:
                                        end_date = date + pd.Timedelta(days=days)
                                        for bs_date in black_swan_dates:
                                            if date <= bs_date <= end_date:
                                                count += 1
                                                break
                                    black_swan_counts[pattern].append(count)

                            # 결과 시각화
                            fig, ax = plt.subplots(figsize=(8, 6))
                            width = 0.15
                            x = np.arange(len(pattern_types))

                            for i, days in enumerate(days_thresholds):
                                counts = [black_swan_counts[pattern][i] for pattern in pattern_types]
                                ax.bar(x + i * width, counts, width, label=f'Within {days} Days')

                            ax.set_title('Black Swan Events After Pattern', fontsize=16)
                            ax.set_xlabel('Pattern Type', fontsize=14)
                            ax.set_ylabel('Number of Black Swan Events', fontsize=14)
                            ax.set_xticks(x + width * (len(days_thresholds) - 1) / 2)
                            ax.set_xticklabels(pattern_types, rotation=45, ha='right')
                            ax.legend()
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        # 추가 분석 섹션 (패턴별 상세 분석) - 한 줄에 두 개씩 표시
                        st.write("### Pattern-Specific Analysis")
                        
                        # 패턴 유형 그룹화 (최대 6개까지만 표시)
                        patterns_to_analyze = pattern_types[:min(6, len(pattern_types))]
                        num_patterns = len(patterns_to_analyze)
                        rows_needed = (num_patterns + 1) // 2  # 올림 나눗셈
                        
                        for row in range(rows_needed):
                            # 각 행마다 2개의 컬럼 생성
                            cols = st.columns(2)
                            
                            for col_idx in range(2):
                                pattern_idx = row * 2 + col_idx
                                
                                # 패턴 인덱스가 유효한지 확인
                                if pattern_idx < num_patterns:
                                    pattern = patterns_to_analyze[pattern_idx]
                                    
                                    with cols[col_idx]:
                                        st.subheader(f"{pattern} Pattern Analysis")
                                        
                                        # 해당 패턴의 발생 빈도와 블랙 스완 이벤트 관계
                                        pattern_dates = technical_patterns_df[technical_patterns_df['pattern'] == pattern]['date']
                                        
                                        # 결과 시각화 (파이 차트)
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        
                                        # 이 패턴 발생 후 20일 이내에 블랙 스완 이벤트가 발생한 비율 계산
                                        pattern_occurrences = len(pattern_dates)
                                        followed_by_black_swan = black_swan_counts[pattern][-1]  # 20일 이내 블랙 스완 이벤트
                                        not_followed = pattern_occurrences - followed_by_black_swan
                                        
                                        # 파이 차트 데이터
                                        labels = ['Followed by Black Swan', 'Not Followed by Black Swan']
                                        sizes = [followed_by_black_swan, not_followed]
                                        colors = ['red', 'lightgray']
                                        explode = (0.1, 0)  # 첫 번째 조각만 약간 분리
                                        
                                        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                                            shadow=True, startangle=90)
                                        ax.axis('equal')  # 원형 파이 차트를 위해
                                        
                                        plt.title(f'{pattern}: 20-Day Black Swan Follow-up', fontsize=14)
                                        plt.tight_layout()
                                        
                                        st.pyplot(fig)
                                        plt.close()
                                        
                                        # 패턴 발생 후 일별 누적 블랙 스완 발생 추이
                                        if followed_by_black_swan > 0:
                                            cumulative_counts = [black_swan_counts[pattern][i] for i in range(len(days_thresholds))]
                                            
                                            fig, ax = plt.subplots(figsize=(8, 4))
                                            ax.plot(days_thresholds, cumulative_counts, marker='o', color='red', linewidth=2)
                                            
                                            ax.set_title(f'{pattern}: Cumulative Black Swan Events', fontsize=14)
                                            ax.set_xlabel('Days After Pattern', fontsize=12)
                                            ax.set_ylabel('Number of Events', fontsize=12)
                                            ax.grid(True, alpha=0.3)
                                            
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                            plt.close()

                    except Exception as e:
                        st.error(f"❌ Error analyzing technical patterns and black swan events: {str(e)}")
            except Exception as outer_e:
                st.error(f"❌ Error generating technical pattern section: {str(outer_e)}")
def analyze_extreme_events_impact(df, extreme_events_df, black_swan_df, category=None):
    """
    극단적 이벤트와 블랙 스완 이벤트가 수익률에 미치는 영향을 분석합니다.
    
    Parameters:
    -----------
    df : DataFrame
        원본 데이터프레임
    extreme_events_df : DataFrame
        감지된 극단적 이벤트 데이터프레임
    black_swan_df : DataFrame
        감지된 블랙 스완 이벤트 데이터프레임
    category : str, optional
        분석할 특정 카테고리 (기본값: None)
    """
    st.subheader("Impact Analysis of Extreme Events on Returns")
    
    if df is None or df.empty:
        st.warning("분석할 데이터가 없습니다.")
        return
    
    # 필요한 컬럼 확인
    required_cols = ['date', 'category', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"필요한 컬럼이 없습니다: {missing_cols}")
        return
    
    # 카테고리 필터링
    if category:
        df_filtered = df[df['category'] == category].copy()
    else:
        df_filtered = df.copy()
    
    if len(df_filtered) == 0:
        st.warning("No data after filtering.")
        return
    
    # 수익률 계산
    df_filtered['daily_return'] = df_filtered.groupby('category')['close'].pct_change() * 100
    
    # 1. 블랙 스완 이벤트 전후 누적 수익률 분석
    if black_swan_df is not None and not black_swan_df.empty:
        st.write("### Cumulative Returns Before and After Black Swan Events")
        
        try:
            # 카테고리 및 날짜별로 데이터 그룹화
            grouped = df_filtered.groupby(['category', 'date'])['daily_return'].mean().reset_index()
            
            # 카테고리 선택
            categories_to_analyze = [category] if category else grouped['category'].unique()[:min(4, len(grouped['category'].unique()))]
            
            for cat in categories_to_analyze:
                # 해당 카테고리의 블랙 스완 이벤트 필터링
                cat_black_swans = black_swan_df[(black_swan_df['category'] == cat) | 
                                              (black_swan_df['category'] == 'Economic Indicator')]
                
                if cat_black_swans.empty:
                    st.info(f"There are no black swan events available for the {cat} category.")
                    continue
                
                # 해당 카테고리의 데이터
                cat_data = grouped[grouped['category'] == cat].sort_values('date')
                
                # 각 블랙 스완 이벤트에 대해 전후 20일 간의 누적 수익률 계산
                window_size = 20
                
                # 각 카테고리마다 새로운 figure 생성
                fig, ax = plt.subplots(figsize=(14, 8))
                
                for i, (_, event) in enumerate(cat_black_swans.head(5).iterrows()):  # 최대 5개 이벤트
                    event_date = event['date']
                    event_type = event['event_type']
                    
                    # 이벤트 전후 데이터 선택
                    mask = (cat_data['date'] >= event_date - pd.Timedelta(days=window_size)) & \
                           (cat_data['date'] <= event_date + pd.Timedelta(days=window_size))
                    event_window_data = cat_data.loc[mask].copy()
                    
                    if len(event_window_data) > 1:  # 충분한 데이터가 있는 경우
                        # 이벤트 날짜를 0으로 조정
                        event_window_data['days_from_event'] = (event_window_data['date'] - event_date).dt.days
                        
                        # 누적 수익률 계산 (이벤트 날짜를 기준으로 복리화)
                        event_window_data['cum_return'] = 100 * ((1 + event_window_data['daily_return']/100).cumprod() - 1)
                        
                        # 이벤트 날짜 기준으로 재조정
                        try: 
                            event_idx = event_window_data[event_window_data['days_from_event'] == 0].index
                        
                            if len(event_idx) > 0:
                                if 'cum_return' not in event_window_data.columns:
                                    st.warning("No cumulative return data available")
                                    continue
                                    
                                event_return = event_window_data.loc[event_idx[0], 'cum_return']
                                event_window_data['adjusted_cum_return'] = event_window_data['cum_return'] - event_return
                                
                                # 플롯
                                ax.plot(
                                    event_window_data['days_from_event'], 
                                    event_window_data['adjusted_cum_return'],
                                    label=f"{event_date.strftime('%Y-%m-%d')}: {event_type}",
                                    marker='o', markersize=3
                                )
                        except Exception as e:
                            st.warning(f"이벤트 수익률 분석 중 오류가 발생했습니다: {str(e)}")
                            continue 
                
                # 이벤트 발생일 표시
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Event Date')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                ax.set_title(f'Cumulative Returns Before and After Black Swan Events for {cat}', fontsize=16)
                ax.set_xlabel('Days Relative to Event', fontsize=14)
                ax.set_ylabel('Cumulative Return (%)', fontsize=14)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()  # 각 카테고리 그래프 후 figure 닫기
                
        except Exception as e:
            st.error(f"누적 수익률 분석 중 오류 발생: {str(e)}")
    
    # 2. 극단적 경제 지표와 카테고리별 수익률의 상관관계
    if extreme_events_df is not None and not extreme_events_df.empty:
        st.write("### Correlation Between Extreme Economic Events and Returns")
        
        try:
            # 경제 지표 극단값 필터링
            economic_extremes = extreme_events_df[
                ((extreme_events_df['event_type'] == 'Extreme High') | 
                 (extreme_events_df['event_type'] == 'Extreme Low')) &
                (extreme_events_df['indicator'].str.contains('_norm'))
            ]
            
            if economic_extremes.empty:
                st.info("There are no extreme economic events available for analysis.")
            else:
                # 극단적 이벤트가 발생한 날짜
                extreme_dates = economic_extremes['date'].unique()
                
                # 이벤트 발생 이후 1, 3, 5, 10일 간의 평균 수익률 계산
                holding_periods = [1, 3, 5, 10]
                
                # 카테고리 및 지표별 이벤트 후 수익률
                results = []
                
                for indicator in economic_extremes['indicator'].unique():
                    indicator_name = indicator.split('_')[0]  # 지표명 추출
                    
                    high_events = economic_extremes[
                        (economic_extremes['indicator'] == indicator) & 
                        (economic_extremes['event_type'] == 'Extreme High')
                    ]
                    
                    low_events = economic_extremes[
                        (economic_extremes['indicator'] == indicator) & 
                        (economic_extremes['event_type'] == 'Extreme Low')
                    ]
                    
                    for cat in df_filtered['category'].unique():
                        cat_data = df_filtered[df_filtered['category'] == cat].sort_values('date')
                        
                        # 각 보유 기간에 대해
                        for days in holding_periods:
                            # 극단적 고점 이벤트 후 수익률
                            high_returns = []
                            for event_date in high_events['date']:
                                future_data = cat_data[cat_data['date'] > event_date].head(days)
                                if len(future_data) == days:  # 충분한 미래 데이터가 있는 경우
                                    cum_return = 100 * ((1 + future_data['daily_return']/100).prod() - 1)
                                    high_returns.append(cum_return)
                            
                            high_mean = np.mean(high_returns) if high_returns else np.nan
                            
                            # 극단적 저점 이벤트 후 수익률
                            low_returns = []
                            for event_date in low_events['date']:
                                future_data = cat_data[cat_data['date'] > event_date].head(days)
                                if len(future_data) == days:  # 충분한 미래 데이터가 있는 경우
                                    cum_return = 100 * ((1 + future_data['daily_return']/100).prod() - 1)
                                    low_returns.append(cum_return)
                            
                            low_mean = np.mean(low_returns) if low_returns else np.nan
                            
                            # 결과 저장
                            results.append({
                                'indicator': indicator_name,
                                'category': cat,
                                'holding_period': days,
                                'high_event_return': high_mean,
                                'low_event_return': low_mean,
                                'high_event_count': len(high_returns),
                                'low_event_count': len(low_returns)
                            })
                
                # 결과를 데이터프레임으로 변환
                if results:
                    result_df = pd.DataFrame(results)
                    
                    # 지표별 평균 수익률 시각화
                    for period in holding_periods:
                        period_data = result_df[result_df['holding_period'] == period]
                        
                        if not period_data.empty:
                            fig, ax = plt.subplots(figsize=(14, 8))
                            
                            # 지표별로 그룹화
                            indicator_means = period_data.groupby('indicator').agg({
                                'high_event_return': 'mean',
                                'low_event_return': 'mean'
                            }).reset_index()
                            
                            x = np.arange(len(indicator_means))
                            width = 0.35
                            
                            ax.bar(x - width/2, indicator_means['high_event_return'], width, 
                                   label='Returns After Extreme High Events', color='red', alpha=0.7)
                            ax.bar(x + width/2, indicator_means['low_event_return'], width, 
                                   label='Returns After Extreme Low Events', color='blue', alpha=0.7)
                            
                            ax.set_title(f'Average Returns After Extreme Economic Events - {period}-Day Holding Period', fontsize=16)
                            ax.set_xlabel('Economic Indicator', fontsize=14)
                            ax.set_ylabel('Avg Return (%)', fontsize=14)
                            ax.set_xticks(x)
                            ax.set_xticklabels(indicator_means['indicator'], rotation=45, ha='right')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                    
                    # 결과 테이블 표시
                    st.write("#### Correlation Between Extreme Economic Indicators and Returns")
                    st.dataframe(result_df.round(2))
                    
        except Exception as e:
            st.error(f"극단적 경제 지표 영향 분석 중 오류 발생: {str(e)}")
def create_extreme_enhanced_data(model_data, extreme_events_df, black_swan_df, technical_patterns_df=None, correlation_breakdown_df=None, predictions_data=None):
    """
    극단적 이벤트 특성을 모델 데이터에 통합하고 머신러닝 예측 결과도 추가합니다.
    
    Parameters:
    -----------
    model_data : dict 또는 DataFrame
        카테고리별 모델 데이터 딕셔너리 또는 단일 DataFrame
    extreme_events_df : DataFrame
        감지된 극단적 이벤트 데이터프레임
    black_swan_df : DataFrame
        감지된 블랙 스완 이벤트 데이터프레임
    technical_patterns_df : DataFrame, optional
        감지된 기술적 패턴 데이터프레임
    correlation_breakdown_df : DataFrame, optional
        감지된 상관관계 붕괴 이벤트 데이터프레임
    predictions_data : dict, optional
        머신러닝 예측 결과 (카테고리별 predictions 컬럼 포함한 데이터프레임)
    
    Returns:
    --------
    extreme_enhanced_model_data : dict
        극단적 이벤트 특성과 예측 결과가 추가된 모델 데이터 딕셔너리
    """
    # 입력 데이터 검증
    if model_data is None:
        return model_data
        
    # DataFrame을 딕셔너리로 변환
    if isinstance(model_data, pd.DataFrame):
        if model_data.empty:
            return {}
            
        if 'category' in model_data.columns:
            model_data_dict = {}
            for category in model_data['category'].unique():
                model_data_dict[category] = model_data[model_data['category'] == category].copy()
            model_data = model_data_dict
        else:
            model_data = {'all': model_data.copy()}
    elif isinstance(model_data, dict):
        if len(model_data) == 0:
            return {}
    else:
        st.warning(f"지원되지 않는 model_data 타입: {type(model_data)}")
        return {}
        
    extreme_enhanced_model_data = {}
    
    # 진행 상황 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    categories = list(model_data.keys())
    
    for i, category in enumerate(categories):
        #status_text.text(f"{category} 모델 데이터 향상 중...")
        
        try:
            # 카테고리별 모델 데이터 복사
            df = model_data[category].copy()
            
            # 1. 블랙 스완 이벤트 특성 추가
            if black_swan_df is not None and not black_swan_df.empty:
                category_black_swans = black_swan_df[black_swan_df['category'] == category]
                
                if not category_black_swans.empty:
                    # 블랙 스완 이벤트 발생 여부
                    black_swan_dates = category_black_swans['date'].tolist()
                    df['black_swan_event'] = df.index.isin(black_swan_dates).astype(int)
                    
                    # 최근 블랙 스완 이벤트까지의 거리
                    df['days_since_black_swan'] = 999  # 기본값
                    
                    for date in df.index:
                        date_ts = pd.Timestamp(date) if not isinstance(date, pd.Timestamp) else date
                        
                        # 이 날짜 이전의 모든 블랙 스완
                        prev_black_swans = []
                        for d in black_swan_dates:
                            d_ts = pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d
                            if d_ts <= date_ts:
                                prev_black_swans.append(d_ts)
                        
                        if prev_black_swans:
                            latest_black_swan = max(prev_black_swans)
                            days_diff = (date_ts - latest_black_swan).days
                            df.at[date, 'days_since_black_swan'] = days_diff
                    
                    # 블랙 스완 Z-score 특성
                    df['black_swan_zscore'] = 0
                    
                    for _, event in category_black_swans.iterrows():
                        if event['date'] in df.index:
                            df.at[event['date'], 'black_swan_zscore'] = event['z_score']
            
            # 2. 극단적 경제 지표 이벤트 특성 추가
            if extreme_events_df is not None and not extreme_events_df.empty:
                # 경제 지표 극단값 이벤트
                economic_extremes = extreme_events_df[
                    ((extreme_events_df['event_type'] == 'Extreme High') | 
                    (extreme_events_df['event_type'] == 'Extreme Low')) &
                    (extreme_events_df['indicator'].str.contains('_norm'))
                ]
                
                if not economic_extremes.empty:
                    # 경제 지표 극단 이벤트 발생 여부
                    extreme_dates = economic_extremes['date'].unique()
                    df['extreme_economic_event'] = df.index.isin(extreme_dates).astype(int)
                    
                    # 최근 극단 이벤트까지의 거리
                    df['days_since_extreme_event'] = 999  # 기본값
                    
                    for date in df.index:
                        date_ts = pd.Timestamp(date) if not isinstance(date, pd.Timestamp) else date
                         
                        prev_extreme_event = []
                        for d in extreme_dates:
                            d_ts = pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d
                            if d_ts <= date_ts:
                                prev_extreme_event.append(d_ts)
                        
                        if prev_extreme_event:
                            latest_extreme_event = max(prev_extreme_event)
                            days_diff = (date_ts - latest_extreme_event).days
                            df.at[date, 'days_since_extreme_event'] = days_diff
                    
                    # 주요 경제 지표별 극단 이벤트 특성
                    important_indicators = ['GDP_norm', 'CPI_norm', 'Fed_Funds_Rate_norm', 'Dollar_Index_norm']
                    
                    for indicator in important_indicators:
                        # 극단적 고점 이벤트
                        high_events = economic_extremes[
                            (economic_extremes['indicator'] == indicator) & 
                            (economic_extremes['event_type'] == 'Extreme High')
                        ]
                        
                        # 극단적 저점 이벤트
                        low_events = economic_extremes[
                            (economic_extremes['indicator'] == indicator) & 
                            (economic_extremes['event_type'] == 'Extreme Low')
                        ]
                        
                        # 각 지표별 극단 이벤트 특성 추가
                        ind_name = indicator.split('_')[0]
                        
                        # 고점 이벤트 특성
                        df[f'{ind_name}_extreme_high'] = df.index.isin(high_events['date']).astype(int)
                        
                        # 저점 이벤트 특성
                        df[f'{ind_name}_extreme_low'] = df.index.isin(low_events['date']).astype(int)
            
            # 3. 기술적 패턴 특성 추가
            if technical_patterns_df is not None and not technical_patterns_df.empty:
                # 해당 카테고리의 기술적 패턴
                category_patterns = technical_patterns_df[technical_patterns_df['category'] == category]
                
                if not category_patterns.empty:
                    # 모든 기술적 패턴 발생 여부
                    pattern_dates = category_patterns['date'].unique()
                    df['technical_pattern_event'] = df.index.isin(pattern_dates).astype(int)
                    
                    # 패턴 유형별 특성 추가
                    pattern_types = [
                        'Golden Cross (5-20)', 'Golden Cross (20-50)',
                        'Death Cross (5-20)', 'Death Cross (20-50)',
                        'Uptrend Reversal', 'Downtrend Reversal',
                        'Support Test', 'Resistance Test'
                    ]
                    
                    for pattern in pattern_types:
                        pattern_specific = category_patterns[category_patterns['pattern'] == pattern]
                        
                        if not pattern_specific.empty:
                            # 패턴 발생 여부
                            pattern_col_name = pattern.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()
                            df[f'pattern_{pattern_col_name}'] = df.index.isin(pattern_specific['date']).astype(int)
            
            # 4. 상관관계 붕괴 이벤트 특성 추가
            if correlation_breakdown_df is not None and not correlation_breakdown_df.empty:
                # 해당 카테고리가 포함된 상관관계 붕괴 이벤트
                category_correlations = correlation_breakdown_df[correlation_breakdown_df['pair'].str.contains(category)]
                
                if not category_correlations.empty:
                    # 상관관계 붕괴 이벤트 발생 여부
                    correlation_dates = category_correlations['date'].unique()
                    df['correlation_breakdown_event'] = df.index.isin(correlation_dates).astype(int)
                    
                    # 최근 상관관계 붕괴 이벤트까지의 거리
                    df['days_since_correlation_breakdown'] = 999  # 기본값
                    
                    for date in df.index:
                        date_ts = pd.Timestamp(date) if not isinstance(date, pd.Timestamp) else date
                        
                        prev_breakdowns = []
                        for d in correlation_dates:
                            d_ts = pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d
                            if d_ts <= date_ts:
                                prev_breakdowns.append(d_ts)
                    
                        if prev_breakdowns:
                            latest_breakdown = max(prev_breakdowns)
                            days_diff = (date_ts - latest_breakdown).days
                            df.at[date, 'days_since_correlation_breakdown'] = days_diff
            
            # 5. 머신러닝 예측 결과 추가 - 개선된 버전
            if predictions_data is not None and category in predictions_data:
                pred_df = predictions_data[category]
                
                if isinstance(pred_df, pd.DataFrame) and 'predictions' in pred_df.columns:
                    # 예측 컬럼이 아직 df에 없으면 생성
                    if 'predictions' not in df.columns:
                        df['predictions'] = np.nan
                    
                    # 인덱스 처리 통합 로직
                    try:
                        # 1. 날짜 형식 인덱스를 사용하는 경우 - 직접 매핑
                        if (isinstance(df.index[0], (pd.Timestamp, np.datetime64, datetime.date)) if len(df.index) > 0 else False) and \
                           (isinstance(pred_df.index[0], (pd.Timestamp, np.datetime64, datetime.date)) if len(pred_df.index) > 0 else False):
                            
                            # 인덱스 기반 예측값 복사
                            common_indices = set(df.index) & set(pred_df.index)
                            for idx in common_indices:
                                if pd.notna(pred_df.loc[idx, 'predictions']):
                                    df.at[idx, 'predictions'] = pred_df.loc[idx, 'predictions']
                        
                        # 2. 날짜 컬럼 기반 병합 필요한 경우
                        else:
                            # 인덱스를 컬럼으로 변환
                            df_reset = df.reset_index()
                            pred_reset = pred_df.reset_index()
                            
                            # 날짜 컬럼 찾기
                            date_cols = ['date', 'datetime', 'time', 'index', 'timestamp']
                            
                            date_col_model = next((col for col in df_reset.columns if col.lower() in date_cols), df_reset.columns[0])
                            date_col_pred = next((col for col in pred_reset.columns if col.lower() in date_cols), pred_reset.columns[0])
                            
                            # 날짜 형식으로 변환
                            df_reset[date_col_model] = pd.to_datetime(df_reset[date_col_model], errors='coerce').dt.date
                            pred_reset[date_col_pred] = pd.to_datetime(pred_reset[date_col_pred], errors='coerce').dt.date
                            
                            # null 날짜 제거
                            df_reset = df_reset.dropna(subset=[date_col_model])
                            pred_reset = pred_reset.dropna(subset=[date_col_pred])
                            
                            # 컬럼 이름이 다르면 통일
                            if date_col_pred != date_col_model:
                                pred_reset = pred_reset.rename(columns={date_col_pred: date_col_model})
                            
                            # 병합
                            merged_df = pd.merge(
                                df_reset,
                                pred_reset[[date_col_model, 'predictions']],
                                on=date_col_model,
                                how='left',
                                suffixes=('', '_pred')
                            )
                            
                            # 예측 값 통합
                            if 'predictions_pred' in merged_df.columns:
                                merged_df['predictions'] = merged_df['predictions'].fillna(merged_df['predictions_pred'])
                                merged_df = merged_df.drop('predictions_pred', axis=1)
                            
                            # 인덱스 복원 (날짜로)
                            merged_df = merged_df.set_index(date_col_model)
                            df = merged_df
                    
                    except Exception as e:
                        st.warning(f"{category}의 예측값 병합 중 오류 발생: {str(e)}")
            
            # 향상된 모델 데이터 저장
            extreme_enhanced_model_data[category] = df
            
        except Exception as e:
            st.error(f"{category} 처리 중 오류 발생: {str(e)}")
        
        # 진행 상황 업데이트
        progress_bar.progress((i + 1) / len(categories))
    
    #status_text.text("모델 데이터 향상 완료!")
    progress_bar.empty()
    
    return extreme_enhanced_model_data

                            
import pandas as pd
import numpy as np
from datetime import datetime
def generate_executive_summary(extreme_events_df, black_swan_df, technical_patterns_df, correlation_breakdown_df, 
                              daily_avg_prices, daily_indicators, extreme_enhanced_model_data, categories, precalculated_predictions=None):
    """
    데이터에 기반한 투자 요약 보고서를 생성합니다.
    
    Parameters:
    -----------
    extreme_events_df : pandas.DataFrame
        극단적 시장 이벤트 데이터
    black_swan_df : pandas.DataFrame
        블랙 스완 이벤트 데이터
    technical_patterns_df : pandas.DataFrame
        기술적 패턴 데이터
    correlation_breakdown_df : pandas.DataFrame
        상관관계 붕괴 데이터
    daily_avg_prices : pandas.DataFrame
        일별 평균 가격 데이터
    daily_indicators : pandas.DataFrame
        일별 지표 데이터
    extreme_enhanced_model_data : dict
        카테고리별 모델 예측 데이터 딕셔너리
    categories : list
        분석 카테고리 목록
    
    Returns:
    --------
    dict
        투자 요약 보고서 딕셔너리
    """
    summary = {}
    
    # 이벤트 카운트 계산
    extreme_count = len(extreme_events_df) if extreme_events_df is not None and not extreme_events_df.empty else 0
    black_swan_count = len(black_swan_df) if black_swan_df is not None and not black_swan_df.empty else 0
    pattern_count = len(technical_patterns_df) if technical_patterns_df is not None and not technical_patterns_df.empty else 0
    correlation_count = len(correlation_breakdown_df) if correlation_breakdown_df is not None and not correlation_breakdown_df.empty else 0

    total_events = extreme_count + black_swan_count + pattern_count + correlation_count
    
    # 리스크 레벨 결정
    if total_events == 0:
        risk_level = "Low Risk"
        risk_score = 1
    elif total_events < 10:
        risk_level = "Caution"
        risk_score = 2
    elif total_events < 30:
        risk_level = "Alert"
        risk_score = 3
    else:
        risk_level = "High Risk"
        risk_score = 4
    
    # 카테고리별 영향도 분석
    category_impact = {}
    
    for category in categories:
        impact_score = 0

        # 블랙 스완 영향
        if black_swan_df is not None and not black_swan_df.empty and 'category' in black_swan_df.columns:
            category_black_swans = black_swan_df[black_swan_df['category'] == category]
            impact_score += len(category_black_swans) * 3
        
        # 기술적 패턴 영향
        if technical_patterns_df is not None and not technical_patterns_df.empty and 'category' in technical_patterns_df.columns:
            category_patterns = technical_patterns_df[technical_patterns_df['category'] == category]
            impact_score += len(category_patterns)
        
        # 극단적 이벤트 영향
        if extreme_events_df is not None and not extreme_events_df.empty and 'indicator' in extreme_events_df.columns:
            category_extremes = extreme_events_df[extreme_events_df['indicator'].str.contains(category, na=False)]
            impact_score += len(category_extremes) * 2
        
        category_impact[category] = impact_score
    
    # 가장 영향을 많이/적게 받은 카테고리 식별
    if category_impact:
        most_impacted = max(category_impact, key=category_impact.get)
        least_impacted = min(category_impact, key=category_impact.get)
    else:
        most_impacted = "N/A"
        least_impacted = "N/A"

    # 주요 발견사항 분석
    key_findings = []
    
    # 블랙 스완 이벤트 분석
    if black_swan_df is not None and not black_swan_df.empty:
        try:
            recent_black_swan = black_swan_df.sort_values('date', ascending=False).iloc[0]
            date_str = recent_black_swan['date'].strftime('%Y-%m-%d') if pd.notnull(recent_black_swan['date']) else "N/A"
            key_findings.append(
                f"🔥 Latest black swan event occurred in {recent_black_swan.get('category', 'N/A')} as a "
                f"{recent_black_swan.get('event_type', 'N/A')} on ({date_str})"
            )
        except Exception:
            pass
    
    # 기술적 패턴 분석
    if technical_patterns_df is not None and not technical_patterns_df.empty and 'pattern' in technical_patterns_df.columns:
        try:
            if len(technical_patterns_df['pattern'].value_counts()) > 0:
                most_common_pattern = technical_patterns_df['pattern'].mode()[0]
                pattern_count = len(technical_patterns_df[technical_patterns_df['pattern'] == most_common_pattern])
                key_findings.append(
                    f"📊 Most common technical pattern: {most_common_pattern} ({pattern_count} times)"
                )
        except Exception:
            pass
    
    # 상관관계 붕괴 분석
    if (correlation_breakdown_df is not None and 
        not correlation_breakdown_df.empty and 
        'change' in correlation_breakdown_df.columns):
        try:
            non_null_change = correlation_breakdown_df['change'].dropna()
            if not non_null_change.empty:
                largest_breakdown_idx = non_null_change.abs().idxmax()
                largest_breakdown = correlation_breakdown_df.loc[largest_breakdown_idx]

                pair = largest_breakdown.get('pair', 'N/A')
                old_corr = largest_breakdown.get('old_correlation', None)
                new_corr = largest_breakdown.get('new_correlation', None)

                if old_corr is not None and new_corr is not None:
                    key_findings.append(
                        f"💔 The largest Correlation Breakedown: {pair} "
                        f"({old_corr:.2f} → {new_corr:.2f})"
                    )
        except Exception:
            pass

    # 극단적 이벤트 분석
    if extreme_events_df is not None and not extreme_events_df.empty and 'indicator' in extreme_events_df.columns:
        try:
            economic_indicators = extreme_events_df[extreme_events_df['indicator'].str.contains('_norm', na=False)]
            indicator_counts = economic_indicators['indicator'].value_counts()
            if len(indicator_counts) > 0:
                most_volatile_indicator = indicator_counts.index[0]
                count = indicator_counts.iloc[0]
                key_findings.append(
                    f"⚡ The most volatile economic indicator is {most_volatile_indicator.replace('_norm', '')}, with {count} extreme events observed."
                )
        except Exception:
            pass
    
    # 최근 30일 변동성 분석
    if daily_avg_prices is not None and not daily_avg_prices.empty:
        try:
            # 최근 30일 데이터 추출
            if isinstance(daily_avg_prices.index, pd.DatetimeIndex):
                last_30_days = daily_avg_prices.last('30D')
            else:
                # 날짜 인덱스가 아닌 경우 마지막 30개 행 사용
                last_30_days = daily_avg_prices.iloc[-30:]
                
            if not last_30_days.empty:
                volatility_30d = last_30_days.pct_change().std() * np.sqrt(252)  # 연율화된 변동성
                if not volatility_30d.isnull().all():
                    most_volatile_category = volatility_30d.idxmax()
                    volatility_value = volatility_30d.max()
                    key_findings.append(
                        f"📈 Category with the highest 30-day volatility: {most_volatile_category} (Annualized Volatility: {volatility_value:.1%})"
                    )
        except Exception:
            pass
    
    # 모델 예측 분석
    if precalculated_predictions and isinstance(precalculated_predictions, dict) and 'bullish' in precalculated_predictions and 'bearish' in precalculated_predictions:
        bullish_categories = precalculated_predictions['bullish']
        bearish_categories = precalculated_predictions['bearish']
        prediction_details = precalculated_predictions.get('details', {})
    else:
        # 자체 예측 분석 코드
        bullish_categories = []
        bearish_categories = []
        prediction_details = {}
        
        if extreme_enhanced_model_data is not None and isinstance(extreme_enhanced_model_data, dict):
            try:
                for category, data in extreme_enhanced_model_data.items():
                    if isinstance(data, pd.DataFrame) and 'predictions' in data.columns:
                        valid_predictions = data['predictions'].dropna()
                        
                        if len(valid_predictions) > 0:
                            # 최근 50개 예측 또는 가능한 최대 수
                            recent_predictions = valid_predictions.iloc[-min(50, len(valid_predictions)):]
                            up_ratio = (recent_predictions == 1).mean()
                            
                            # 예측 세부 정보 저장
                            prediction_details[category] = {
                                'samples': len(valid_predictions),
                                'bullish_ratio': float(up_ratio),
                                'direction': 'bullish' if up_ratio >= 0.49 else 'bearish' if up_ratio <= 0.45 else 'neutral'
                            }
                            
                            # 분류 기준 완화 (55%+ 상승 = bullish, 45%- 상승 = bearish)
                            if up_ratio >= 0.49:
                                bullish_categories.append(category)
                            elif up_ratio <= 0.45:
                                bearish_categories.append(category)
            except Exception:
                pass
    
    # 모델 예측 결과를 key_findings에 추가
    if bullish_categories:
        key_findings.append(f"🚀 Bullish forecast: {', '.join(bullish_categories)}")
    if bearish_categories:
        key_findings.append(f"⚠️ Bearish forecast: {', '.join(bearish_categories)}")
    
    # 현재 시장 감성 분석
    current_sentiment = None
    if daily_indicators is not None and not daily_indicators.empty:
        try:
            sentiment_cols = [col for col in daily_indicators.columns if 'sentiment' in col or 'fear_greed' in col]
            if sentiment_cols and len(daily_indicators) > 0:
                latest_sentiment = daily_indicators[sentiment_cols].iloc[-1].mean()
                if pd.notnull(latest_sentiment):
                    if latest_sentiment > 0.6:
                        current_sentiment = "Very Positive"
                    elif latest_sentiment > 0.4:
                        current_sentiment = "Positive"
                    elif latest_sentiment > 0.3:
                        current_sentiment = "Neutral"
                    else:
                        current_sentiment = "Negative"
        except Exception:
            pass
    
    # 투자 제안사항 생성      
    investment_suggestions = []
    
    # 리스크 수준에 따른 제안
    if risk_score == 1:
        investment_suggestions.extend([
            "✅ The current market is stable. Consider maintaining your existing positions.",
            "📈 Focus on sectors with high growth potential."
        ])
    elif risk_score == 2:
        investment_suggestions.extend([
            "⚠️ Caution signals detected in the market. Strengthen your risk management strategies.",
            "🛡️ Consider building hedge positions.",
            f"📉 Review your exposure to the {most_impacted} sector."
        ])
    elif risk_score == 3:
        investment_suggestions.extend([
            "⚠️⚠️ Market is in an alert state. A more conservative approach is recommended.",
            "💰 Consider increasing cash holdings.",
            f"🔍 Reduce exposure to the {most_impacted} sector and consider rotating into {least_impacted}."
        ])
    else:  # risk_score == 4
        investment_suggestions.extend([
            "🚨 The market is at high risk. Extreme caution is advised.",
            "🏦 Consider shifting into safer assets.",
            f"❌ It is recommended to avoid the {most_impacted} sector.",
            "📊 Plan to re-enter the market after stabilization."
        ])
    
    # 시장 감성 정보 추가
    if current_sentiment:
        extra_caution = " — Additional caution advised." if risk_score > 2 else ""
        investment_suggestions.append(f"🎯 Current market sentiment: {current_sentiment}{extra_caution}")
    
    # 모델 기반 추가 제안
    if bullish_categories:
        investment_suggestions.append(f"🎯 AI model suggests bullish trend in: {', '.join(bullish_categories)}")

    if bearish_categories:
        investment_suggestions.append(f"⚠️ AI model flags bearish trend in: {', '.join(bearish_categories)}")
    
    # 추가 경고
    if black_swan_count > 3:
        investment_suggestions.append("🚨 Multiple black swan events detected recently. Extreme caution is recommended.")
    
    # 최종 요약 생성
    summary = {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'total_events': total_events,
        'most_impacted': most_impacted,
        'least_impacted': least_impacted,
        'key_findings': key_findings,
        'investment_suggestions': investment_suggestions,
        'event_counts': {
            'black_swan': black_swan_count,
            'extreme': extreme_count,
            'pattern': pattern_count,
            'correlation': correlation_count
        },
        'current_sentiment': current_sentiment,
        'model_predictions': {
            'bullish': bullish_categories,
            'bearish': bearish_categories,
            'details': prediction_details  # 예측 세부 정보 추가
        }
    }

    return summary

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import traceback
from matplotlib.patches import Patch

def fix_feature_importance(model_results):
    """
    Fixes and standardizes feature importance data in model results.
    Instead of skipping invalid entries, this function converts or repairs them.
    
    Parameters:
    -----------
    model_results : dict
        Original model results data
        
    Returns:
    --------
    dict
        Model results with standardized feature importance data
    """
    if not isinstance(model_results, dict):
        return {}
    
    fixed_results = {}
    
    for category, result in model_results.items():
        if not result or not isinstance(result, dict):
            continue
            
        # Copy the result to avoid modifying original
        fixed_result = result.copy()
        
        # Fix feature_importance if it exists
        if 'feature_importance' in fixed_result:
            feature_importance = fixed_result['feature_importance']
            
            # Case 1: If feature_importance is a float/number (common error case)
            if isinstance(feature_importance, (float, int, np.float64, np.int64)):
                # Convert to a dictionary with a single "overall" feature
                fixed_result['feature_importance'] = {
                    "overall_importance": float(feature_importance)
                }
            
            # Case 2: If feature_importance is None or empty
            elif feature_importance is None or feature_importance == {}:
                # Create an empty dictionary
                fixed_result['feature_importance'] = {}
            
            # Case 3: If feature_importance is already a dictionary, validate its values
            elif isinstance(feature_importance, dict):
                # Validate and convert all values to float
                valid_features = {}
                for k, v in feature_importance.items():
                    try:
                        value = float(v)
                        if pd.notna(value) and np.isfinite(value):
                            valid_features[k] = value
                    except (ValueError, TypeError):
                        # Skip this feature if value can't be converted to float
                        pass
                        
                fixed_result['feature_importance'] = valid_features
            
            # Case 4: Any other type, create an empty dictionary
            else:
                fixed_result['feature_importance'] = {}
                
        # If feature_importance doesn't exist, add an empty one
        else:
            fixed_result['feature_importance'] = {}
            
        fixed_results[category] = fixed_result
    
    return fixed_results

def is_valid_dataframe(df):
    """Checks if df is a valid DataFrame"""
    return isinstance(df, pd.DataFrame) and not df.empty

def classify_feature_type(feature_name):
    """Classifies feature type based on feature name"""
    if any(eco in feature_name for eco in ["GDP", "CPI", "Dollar", "Fed", "Interest", "Industrial", 
                                        "Treasury", "WTI", "Natural_Gas", "Consumer", "Gov_Spending"]):
        return "Economic"
    elif any(sent in feature_name for sent in ["sentiment", "fear", "greed", "positive", "negative", "neutral"]):
        return "Sentiment"
    elif any(tech in feature_name for tech in ["lag", "rolling", "mean", "MA", "RSI", "MACD", "volatility", 
                                            "change", "acceleration", "cross", "std"]):
        return "Technical"
    elif "corr" in feature_name:
        return "Correlation"
    else:
        return "Other"

def calculate_feature_type_importance(features):
    """Calculates importance by feature type"""
    if not isinstance(features, dict):
        return {}
    
    feature_types = {
        "Economic": [],
        "Sentiment": [],
        "Technical": [],
        "Correlation": [],
        "Other": []
    }
    
    # Classify each feature by type
    for feature in features.keys():
        feature_type = classify_feature_type(feature)
        feature_types[feature_type].append(feature)
    
    # Calculate sum of importance by type
    type_importance = {
        feature_type: sum(features.get(f, 0) for f in feature_list)
        for feature_type, feature_list in feature_types.items()
    }
    
    return type_importance

# --- 메인 대시보드 함수 ---


def generate_focused_ml_insights(model_results, model_data, daily_avg_prices, daily_indicators):
    """
    생성된 ML 모델의 통합 인사이트를 표시합니다.
    각 카테고리별 탭을 생성하고 모델 결과를 보여줍니다.
    
    Args:
        model_results: 카테고리별 모델 결과 딕셔너리
        model_data: 모델이 학습한 데이터
        daily_avg_prices: 일별 평균 가격 데이터
        daily_indicators: 일별 지표 데이터
    """
    st.header("🤖 Machine Learning Model Results")
    
    # 세션에 카테고리별 분석 결과 저장
    if 'category_analyses' not in st.session_state:
        st.session_state['category_analyses'] = {}
    
    # 전체 성능 데이터프레임 생성
    performance_data = []
    for category, result in model_results.items():
        if result:  # 결과가 있는 경우만 처리
            # 최근 예측 확인 - model_data에서 해당 카테고리의 predictions 확인
            latest_prediction = "Unknown"
            if category in model_data and 'predictions' in model_data[category].columns:
                recent_preds = model_data[category]['predictions'].dropna()
                if not recent_preds.empty:
                    latest_pred_val = recent_preds.iloc[-1]
                    latest_prediction = "Up" if latest_pred_val == 1 else "Down"
            
            performance_data.append({
                "Category": category,
                "Model Type": result.get('model_name', 'Unknown'),
                "Accuracy": result.get('accuracy', 0),
                "Optimal Threshold Accuracy": result.get('accuracy_optimal', 0),
                "ROC-AUC": result.get('roc_auc', 0),
                "Prediction Confidence": result.get('roc_auc', 0) * 0.5 + result.get('accuracy_optimal', 0) * 0.5,
                "Latest Prediction": latest_prediction
            })
            
            # 결과를 세션에 저장
            st.session_state['category_analyses'][category] = {
                'model_result': result,
                'performance': {
                    "Model Type": result.get('model_name', 'Unknown'),
                    "Accuracy": result.get('accuracy', 0),
                    "Optimal Threshold Accuracy": result.get('accuracy_optimal', 0),
                    "ROC-AUC": result.get('roc_auc', 0),
                    "Optimal Threshold": result.get('optimal_threshold', 0.5)
                }
            }
    
    # 성능 데이터프레임 생성 및 세션 저장
    performance_df = pd.DataFrame(performance_data)
    st.session_state['full_performance'] = performance_df
    
    # 성능 요약 표시
    st.subheader("📊 Model Performance Overview")
    if not performance_df.empty:
        st.dataframe(performance_df.style.format({
            "Accuracy": "{:.2%}",
            "Optimal Threshold Accuracy": "{:.2%}",
            "ROC-AUC": "{:.2%}",
            "Prediction Confidence": "{:.2%}"
        }))
        
        # 성능 비교 차트
        st.subheader("📈 Model Performance Comparison")
        perf_fig = create_performance_comparison_chart(performance_df)
        st.plotly_chart(perf_fig, use_container_width=True)
    else:
        st.info("No model performance data available yet. Run the analysis first.")
    
    # 카테고리별 탭 생성
    categories = list(model_results.keys())
    if categories:
        tabs = st.tabs([f"📊 {category}" for category in categories])
        
        # 각 카테고리별 탭 내용 설정
        for i, category in enumerate(categories):
            with tabs[i]:
                if category in model_results and model_results[category]:
                    result = model_results[category]
                    
                    # 모델 정보 및 성능 지표
                    st.subheader(f"{category} Category Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model Type", result.get('model_name', 'Unknown'))
                    with col2:
                        st.metric("Accuracy", f"{result.get('accuracy', 0):.2%}")
                    with col3:
                        st.metric("ROC-AUC", f"{result.get('roc_auc', 0):.2%}")
                    
                    # 시각화 섹션
                    st.subheader("📈 Model Visualizations")
                    
                    # 시각화 표시를 위한 그리드
                    col1, col2 = st.columns(2)
                    
                    # 시각화 데이터 접근 및 표시
                    with col1:
                        # 특성 중요도
                        st.markdown("### 🎯 Feature Importance")
                        if 'visualizations' in st.session_state and f"{category}_feature_importance" in st.session_state['visualizations']:
                            st.pyplot(st.session_state['visualizations'][f"{category}_feature_importance"])
                        else:
                            # 파일에서 이미지 로드 시도
                            feature_img_path = os.path.join("ML_results", 'categories', category, 'feature_importance.png')
                            if os.path.exists(feature_img_path):
                                st.image(feature_img_path)
                            else:
                                st.info("Feature importance visualization not available")
                        
                        # ROC 곡선
                        st.markdown("### 📈 ROC Curve")
                        if 'visualizations' in st.session_state and f"{category}roc_curve" in st.session_state['visualizations']:
                            st.pyplot(st.session_state['visualizations'][f"{category}roc_curve"])
                        else:
                            # 파일에서 이미지 로드 시도
                            roc_img_path = os.path.join("ML_results", 'categories', category, 'roc_curve.png')
                            if os.path.exists(roc_img_path):
                                st.image(roc_img_path)
                            else:
                                st.info("ROC curve visualization not available")
                    
                    with col2:
                        # 혼동 행렬
                        st.markdown("### 🧩 Confusion Matrix")
                        if 'visualizations' in st.session_state and f"{category}_Confusion Matrix" in st.session_state['visualizations']:
                            st.pyplot(st.session_state['visualizations'][f"{category}_Confusion Matrix"])
                        else:
                            # 파일에서 이미지 로드 시도
                            cm_img_path = os.path.join("ML_results", 'categories', category, 'confusion_matrix.png')
                            if os.path.exists(cm_img_path):
                                st.image(cm_img_path)
                            else:
                                st.info("Confusion matrix visualization not available")
                        
                        # 확률 분포
                        st.markdown("### 🧮 Prediction Probability")
                        if 'visualizations' in st.session_state and f"{category}probability_distribution" in st.session_state['visualizations']:
                            st.pyplot(st.session_state['visualizations'][f"{category}probability_distribution"])
                        else:
                            # 파일에서 이미지 로드 시도
                            prob_img_path = os.path.join("ML_results", 'categories', category, 'probability_distribution.png')
                            if os.path.exists(prob_img_path):
                                st.image(prob_img_path)
                            else:
                                st.info("Probability distribution visualization not available")
                    
                    # 시간적 예측 패턴
                    st.markdown("### ⏳ Temporal Prediction Trend")
                    if 'visualizations' in st.session_state and f"{category}temporal_prediction" in st.session_state['visualizations']:
                        st.pyplot(st.session_state['visualizations'][f"{category}temporal_prediction"])
                    else:
                        # 파일에서 이미지 로드 시도
                        temporal_img_path = os.path.join("ML_results", 'categories', category, 'temporal_prediction.png')
                        if os.path.exists(temporal_img_path):
                            st.image(temporal_img_path)
                        else:
                            st.info("Temporal prediction visualization not available")
                    
                    # 모델 성능 분석 및 인사이트
                    st.subheader("🔍 Performance Analysis & Insights")
                    
                    # 중요 특성 분석
                    if 'feature_importance' in result and result['feature_importance']:
                        feature_importance = pd.Series(result['feature_importance'])
                        top_features = feature_importance.nlargest(min(5, len(feature_importance)))
                        
                        # 세션에 중요 특성 저장
                        if category in st.session_state['category_analyses']:
                            st.session_state['category_analyses'][category]['top_features'] = top_features.to_dict()
                        
                        st.write("#### 🔑 Key Driving Factors")
                        st.write(f"The most important factors for predicting {category} price movement are:")
                        
                        # 중요 특성 해석
                        st.write("#### 💡 Interpretation")
                        st.write(f"The model for {category} relies heavily on:")
                        for i, (feature, importance) in enumerate(top_features.items()):
                            st.write(f"{i+1}. **{feature}** (Importance: {importance:.3f})")
                        
                        # 모델 성능 및 신뢰도
                        st.write("#### 🎯 Model Reliability")
                        confidence = result.get('roc_auc', 0) * 0.5 + result.get('accuracy_optimal', 0) * 0.5
                        
                        # 세션에 신뢰도 저장
                        if category in st.session_state['category_analyses']:
                            st.session_state['category_analyses'][category]['reliability'] = {
                                'score': float(confidence),
                                'level': 'High' if confidence >= 0.8 else ('Moderate' if confidence >= 0.65 else 'Low')
                            }
                        
                        if confidence >= 0.8:
                            reliability = "High"
                            color = "green"
                        elif confidence >= 0.65:
                            reliability = "Moderate"
                            color = "orange"
                        else:
                            reliability = "Low"
                            color = "red"
                        
                        st.markdown(f"Model reliability: <span style='color:{color};font-weight:bold'>{reliability}</span> ({confidence:.2%})", unsafe_allow_html=True)
                        
                        # 최적 임계값 정보
                        st.write(f"The optimal threshold for predictions is **{result.get('optimal_threshold', 0.5):.3f}** " +
                                f"(vs. standard 0.5), improving accuracy from {result.get('accuracy', 0):.2%} to {result.get('accuracy_optimal', 0):.2%}.")
                else:
                    st.warning(f"No model results available for {category}")
        
        # 예측 결과 표시
        st.subheader("🔮 Predictions Summary")
        if not performance_df.empty:
            # 예측 결과 테이블
            st.dataframe(
                performance_df[['Category', 'Latest Prediction', 'Prediction Confidence']],
                column_config={
                    "Category": st.column_config.TextColumn("Category"),
                    "Latest Prediction": st.column_config.TextColumn("Latest Prediction", help="Most recent prediction (Up/Down)"),
                    "Prediction Confidence": st.column_config.ProgressColumn(
                        "Prediction Confidence",
                        format="%.1f%%",
                        min_value=0,
                        max_value=1,
                    )
                }
            )
            
            # 예측 패널 생성
            st.markdown("### 🎯 Current Predictions")
            prediction_cols = st.columns(min(4, len(categories)))
            
            for i, category in enumerate(categories[:min(4, len(categories))]):
                with prediction_cols[i % len(prediction_cols)]:
                    if category in performance_df['Category'].values:
                        pred_row = performance_df[performance_df['Category'] == category].iloc[0]
                        pred_text = pred_row['Latest Prediction']
                        conf_value = pred_row['Prediction Confidence']
                        
                        # 예측 방향에 따른 색상 및 아이콘
                        if pred_text == "Up":
                            color = "green"
                            icon = "📈"
                        elif pred_text == "Down":
                            color = "red"
                            icon = "📉"
                        else:
                            color = "gray"
                            icon = "❓"
                        
                        # 카드 형태로 표시
                        st.markdown(f"""
                        <div style="padding: 15px; border-radius: 5px; background-color: {color}0F; border: 1px solid {color};">
                            <h3 style="color: {color}; margin-top: 0;">{category} {icon}</h3>
                            <p style="font-size: 24px; font-weight: bold; color: {color}; margin: 5px 0;">{pred_text}</p>
                            <p style="color: #666; margin-bottom: 0;">Confidence: {conf_value:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info("No model results available yet. Run the analysis first.")
    
    # 현재 선택된 카테고리 업데이트
    st.session_state['current_category'] = categories[0] if categories else None


# --- 카테고리별 분석 함수 ---
def show_category_analysis(filtered_df, fixed_model_results, cat_name, model_data, daily_avg_prices, daily_indicators):
    try:
        result = fixed_model_results.get(cat_name)
        if not result:
            st.warning("No model result available.")
            return

        st.metric("Accuracy", f"{filtered_df.iloc[0]['Accuracy']:.2%}")
        st.metric("ROC-AUC", f"{filtered_df.iloc[0]['ROC-AUC']:.2%}")
        st.metric("Prediction Confidence", f"{filtered_df.iloc[0]['Prediction Confidence']:.2%}")

        features = result.get('feature_importance', {})
        if features:
            top_features = dict(sorted(features.items(), key=lambda x: x[1], reverse=True)[:10])
            feature_df = pd.DataFrame({"Feature": list(top_features.keys()), "Importance": list(top_features.values())})
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(feature_df['Feature'], feature_df['Importance'], color="skyblue")
            ax.invert_yaxis()
            ax.set_xlabel("Importance")
            ax.set_title(f"Top 10 Important Features: {cat_name}")
            st.pyplot(fig)
            plt.close()

            feature_type_importance = calculate_feature_type_importance(features)
            non_zero_types = {k: v for k, v in feature_type_importance.items() if v > 0}
            if non_zero_types:
                fig, ax = plt.subplots(figsize=(6, 6))
                wedges, texts, autotexts = ax.pie(
                    non_zero_types.values(),
                    labels=non_zero_types.keys(),
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=["salmon", "lightgreen", "lightblue", "purple", "gray"]
                )
                ax.set_title(f"Feature Type Distribution: {cat_name}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            top3_features = dict(sorted(features.items(), key=lambda x: x[1], reverse=True)[:3])
            for feature_name in top3_features:
                feature_series = None
                price_series = None
                
                # Try to find feature in daily_indicators
                if daily_indicators is not None and feature_name in daily_indicators.columns:
                    feature_series = daily_indicators[feature_name]
                
                # Try to find price series in daily_avg_prices
                if daily_avg_prices is not None and cat_name in daily_avg_prices.columns:
                    price_series = daily_avg_prices[cat_name]
                
                if feature_series is not None and price_series is not None:
                    common_index = feature_series.index.intersection(price_series.index)
                    if len(common_index) > 30:
                        fig, ax1 = plt.subplots(figsize=(10, 5))
                        ax1.set_title(f"{feature_name} vs {cat_name} Price")
                        ax1.plot(feature_series[common_index], color='blue', label=feature_name)
                        ax1.set_ylabel(feature_name, color='blue')
                        ax2 = ax1.twinx()
                        ax2.plot(price_series[common_index], color='red', label=f"{cat_name} Price")
                        ax2.set_ylabel(f"{cat_name} Price", color='red')
                        fig.autofmt_xdate()
                        st.pyplot(fig)
                        plt.close()
    except Exception as e:
        st.error(f"Error analyzing {cat_name}: {e}")
        st.error(traceback.format_exc())

# --- Investment Summary ---
def show_investment_and_summary(performance_df):
    try:
        if performance_df.empty:
            st.info("No performance data for investment insights.")
            return
        
        st.subheader("Model-Based Investment Strategy Insights")
        high_conf = performance_df[performance_df['Prediction Confidence'] >= 0.6]
        med_conf = performance_df[(performance_df['Prediction Confidence'] >= 0.5) & (performance_df['Prediction Confidence'] < 0.6)]

        if not high_conf.empty or not med_conf.empty:
            st.write("### 📈 Investment Portfolio Suggestions Based on Confidence")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**High Confidence Categories (Over 60%):**")
                if not high_conf.empty:
                    for _, row in high_conf.iterrows():
                        cat = row['Category']
                        prediction = row['Latest Prediction']
                        arrow = "⬆️" if prediction == "Up" else "⬇️" if prediction == "Down" else "➡️"
                        st.markdown(f"- **{cat}** {arrow}")
                else: 
                    st.info("No high confidence categories.")

                st.markdown("**Medium Confidence Categories (50-60%):**")
                if not med_conf.empty:
                    for _, row in med_conf.iterrows():
                        cat = row['Category']
                        prediction = row['Latest Prediction']
                        arrow = "⬆️" if prediction == "Up" else "⬇️" if prediction == "Down" else "➡️"
                        st.markdown(f"- **{cat}** {arrow}")
                else:
                    st.info("No medium confidence categories.")
            
            with col2:
                # 1. Fixed_model_results preparation
                if 'fixed_model_results' not in st.session_state:
                    if 'model_results' in st.session_state:
                        fixed_model_results = fix_feature_importance(st.session_state['model_results'])
                        st.session_state['fixed_model_results'] = fixed_model_results
                    else:
                        st.error("No model results available.")
                        return
                else:
                    fixed_model_results = st.session_state['fixed_model_results']

                # 3. Create cleaned_type_importance
                if 'cleaned_type_importance' not in st.session_state:
                    cleaned_type_importance = {}
                    for category, result in fixed_model_results.items():
                        features = result.get('feature_importance', {})
                        if features:
                            raw_importance = calculate_feature_type_importance(features)
                            cleaned_importance = {
                                "Economic": raw_importance.get("Economic", 0),
                                "Sentiment": raw_importance.get("Sentiment", 0),
                                "Technical": raw_importance.get("Technical", 0),
                                "Correlation": raw_importance.get("Correlation", 0),
                                "Other": raw_importance.get("Other", 0)
                            }
                            cleaned_type_importance[category] = cleaned_importance
                    st.session_state['cleaned_type_importance'] = cleaned_type_importance
                else:
                    cleaned_type_importance = st.session_state['cleaned_type_importance']

                if cleaned_type_importance:
                    st.markdown("**Prediction Factor-Based Strategies:**")
                    
                    # Economic indicator focus
                    econ_cats = sorted([(cat, imp.get("Economic", 0)) for cat, imp in cleaned_type_importance.items()],
                                       key=lambda x: x[1], reverse=True)[:3]
                    if econ_cats:
                        st.markdown("**Economic Indicator Monitoring:**")
                        for cat, _ in econ_cats:
                            st.markdown(f"- **{cat}**")

                    # Sentiment indicator focus
                    sent_cats = sorted([(cat, imp.get("Sentiment", 0)) for cat, imp in cleaned_type_importance.items()],
                                       key=lambda x: x[1], reverse=True)[:3]
                    if sent_cats:
                        st.markdown("**Market Sentiment Monitoring:**")
                        for cat, _ in sent_cats:
                            st.markdown(f"- **{cat}**")

                    # Technical indicator focus
                    tech_cats = sorted([(cat, imp.get("Technical", 0)) for cat, imp in cleaned_type_importance.items()],
                                       key=lambda x: x[1], reverse=True)[:3]
                    if tech_cats:
                        st.markdown("**Technical Signal Monitoring:**")
                        for cat, _ in tech_cats:
                            st.markdown(f"- **{cat}**")
                else:
                    st.info("Insufficient feature type importance data.")

        # --- 시나리오 제안 ---
        st.write("### 🎯 Machine Learning-Based Investment Scenarios")
        scenarios = []

        # Find high confidence up and down predictions
        high_conf_up = performance_df[(performance_df['Prediction Confidence'] >= 0.6) & 
                                     (performance_df['Latest Prediction'] == "Up")]
        high_conf_down = performance_df[(performance_df['Prediction Confidence'] >= 0.6) & 
                                       (performance_df['Latest Prediction'] == "Down")]

        if not high_conf_up.empty:
            scenarios.append({
                "Title": "Upward Momentum Scenario",
                "Strategy": f"Focus investment on high confidence upward categories: {', '.join(high_conf_up['Category'])}",
                "Risk Level": "Medium",
                "Suitable For": "Aggressive investors"
            })

        if not high_conf_down.empty:
            scenarios.append({
                "Title": "Downside Protection Scenario",
                "Strategy": f"Exclude or hedge categories with high confidence downward predictions: {', '.join(high_conf_down['Category'])}",
                "Risk Level": "Low",
                "Suitable For": "Conservative investors"
            })

        if not scenarios:
            st.info("No clear investment scenarios based on current predictions.")
        else:
            for i, sc in enumerate(scenarios, 1):
                st.markdown(f"**Scenario {i}: {sc['Title']}**")
                st.markdown(f"- **Strategy**: {sc['Strategy']}")
                st.markdown(f"- **Risk Level**: {sc['Risk Level']}")
                st.markdown(f"- **Suitable For**: {sc['Suitable For']}")
                st.write("---")

        # --- Risk warnings ---
        st.write("### ⚠️ Risk Considerations")
        st.markdown("""
        1. **Model Confidence Limitations**
           - Historical data-based models may not adapt perfectly to future market shifts.
           - Even high confidence predictions can be wrong.

        2. **Feature Dependency Risks**
           - Over-reliance on a specific type of indicator (e.g., economic or sentiment) increases vulnerability.
           - Diversified feature importance across types offers more stability.

        3. **Market Environment Changes**
           - Macro-economic changes, policy shifts, or black swan events can override model signals.
           - Regular model retraining and monitoring are essential.
        """)

        # --- Final Executive Summary ---
        st.subheader("Executive Summary")
        avg_accuracy = performance_df['Accuracy'].mean()
        avg_roc_auc = performance_df['ROC-AUC'].mean()
        avg_confidence = performance_df['Prediction Confidence'].mean()

        st.markdown(f"- **Average Accuracy**: {avg_accuracy:.2%}")
        st.markdown(f"- **Average ROC-AUC**: {avg_roc_auc:.2%}")
        st.markdown(f"- **Average Prediction Confidence**: {avg_confidence:.2%}")

    except Exception as e:
        st.error(f"Error generating investment strategy insights: {e}")
        st.error(traceback.format_exc())

def display_executive_dashboard(summary, daily_avg_prices, daily_indicators, black_swan_df, extreme_events_df):
    """
    Displays a comprehensive dashboard.
    """
    st.markdown("### 🎯 Current Market Risk") 
    color_map = {
        "Low Risk": "green",
        "Caution": "orange",
        "Alert": "red",
        "High Risk": "darkred"
    }
    risk_color = color_map.get(summary['risk_level'], "gray")

    st.markdown(f"""
        <div style='background-color: {risk_color}; padding: 20px; border-radius: 10px; text-align: center;'>
            <h1 style='color: white; margin: 0;'>{summary['risk_level']}</h1>
            <p style='color: white; font-size: 20px; margin: 10px 0 0 0;'>Risk Level: {summary['risk_score']}/4</p>            
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Key Market Indicators")

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🦢 Black Swan Events", summary['event_counts']['black_swan'])
    with col2:
        st.metric("⚡ Extreme Events", summary['event_counts']['extreme'])
    with col3:
        st.metric("📈 Technical Patterns", summary['event_counts']['pattern'])
    with col4:
        st.metric("🔄 Correlation Breakdowns", summary['event_counts']['correlation'])

    st.markdown("### 🎯 Category Impact Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style='background-color: #f8d7da; padding: 15px; border-radius: 5px;'>
            <h4 style='color: #721c24; margin: 0;'>Most Impacted Sector</h4>
            <p style='font-size: 24px; font-weight: bold; color: #721c24; margin: 10px 0 0 0;'>{summary['most_impacted']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background-color: #d4edda; padding: 15px; border-radius: 5px;'>
                <h4 style='color: #155724; margin: 0;'>Most Stable Sector</h4>
                <p style='font-size: 24px; font-weight: bold; color: #155724; margin: 10px 0 0 0;'>{summary['least_impacted']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Current market sentiment
    if summary.get('current_sentiment'):
        st.markdown("### 🧠 Market Sentiment Overview")
        sentiment_color = {
            "Very Positive": "green",
            "Positive": "lightgreen",
            "Neutral": "gray",
            "Negative": "red"
        }.get(summary['current_sentiment'], "gray")

        st.markdown(f"""
                    <div style='background-color: {sentiment_color}; padding: 10px; border-radius: 5px; text-align: center;'>
                        <h3 style='color: white; margin: 0;'>{summary['current_sentiment']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
    if summary.get('model_predictions'):
        st.markdown("### 🤖 AI Model Predictions")
        
        # 상승/하락 섹터 구분하여 보기 좋게 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background-color: rgba(0,128,0,0.1); 
                        padding: 15px; 
                        border-radius: 8px; 
                        border-left: 5px solid #4CAF50;'>
                <h4 style='color: #155724; margin-top: 0; margin-bottom: 10px;'>Bullish Sectors</h4>
            """, unsafe_allow_html=True)
            
            if summary['model_predictions']['bullish']:
                sectors_html = ""
                for sector in summary['model_predictions']['bullish']:
                    sectors_html += f"<div style='font-size: 18px; margin: 5px 0;'><span style='color: #155724; font-weight: bold;'>📈 {sector}</span></div>"
                st.markdown(sectors_html, unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #666; font-style: italic;'>No bullish sectors predicted.</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style='background-color: rgba(220,53,69,0.1); 
                        padding: 15px; 
                        border-radius: 8px; 
                        border-left: 5px solid #dc3545;'>
                <h4 style='color: #721c24; margin-top: 0; margin-bottom: 10px;'>Bearish Sectors</h4>
            """, unsafe_allow_html=True)
            
            if summary['model_predictions']['bearish']:
                sectors_html = ""
                for sector in summary['model_predictions']['bearish']:
                    sectors_html += f"<div style='font-size: 18px; margin: 5px 0;'><span style='color: #721c24; font-weight: bold;'>📉 {sector}</span></div>"
                st.markdown(sectors_html, unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #666; font-style: italic;'>No bearish sectors predicted.</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # 예측 세부 정보 테이블 추가 (있는 경우)
        if 'details' in summary['model_predictions'] and summary['model_predictions']['details']:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Prediction Details")
            
            # 테이블 데이터 준비
            table_data = []
            for category, details in summary['model_predictions']['details'].items():
                direction = details.get('direction', 'N/A')
                bullish_ratio = details.get('bullish_ratio', 0)
                
                # 방향에 따라 색상 지정
                color = "#4CAF50" if direction == 'bullish' else "#dc3545" if direction == 'bearish' else "#6c757d"
                icon = "📈" if direction == 'bullish' else "📉" if direction == 'bearish' else "➡️"
                
                # 포맷된 비율 문자열
                ratio_str = f"{bullish_ratio:.2%}"
                
                table_data.append({
                    "Category": category,
                    "Samples": details.get('samples', 0),
                    "Bullish Ratio": ratio_str,
                    "Prediction": f"{icon} {direction.capitalize()}"
                })
            
            if table_data:
                # 데이터프레임 생성
                table_df = pd.DataFrame(table_data)
                
                # 테이블 표시 - 스트림릿 데이터프레임 이용
                st.dataframe(
                    table_df,
                    column_config={
                        "Category": st.column_config.TextColumn("Category", width="medium"),
                        "Samples": st.column_config.NumberColumn("Samples", width="small"),
                        "Bullish Ratio": st.column_config.TextColumn("Bullish Ratio", width="small"),
                        "Prediction": st.column_config.TextColumn("Prediction", width="medium")
                    },
                    use_container_width=True,
                    hide_index=True
                )
     # 주요 발견사항       
    st.markdown("### 🔍 Key Findings")
    for finding in summary['key_findings']:
        st.info(finding)

    # 투자 제안사항   
    st.markdown("### 💼 Investment Recommendations")
    
    for suggestion in summary['investment_suggestions']:
        st.success(suggestion)
    
    # 위험도 추이 차트
    if daily_avg_prices is not None and not daily_avg_prices.empty:
        st.markdown("### 📈 Risk Trend Over Time")
        
        try:
            # datetime 인덱스 확인
            if not pd.api.types.is_datetime64_any_dtype(daily_avg_prices.index):
                daily_avg_prices.index = pd.to_datetime(daily_avg_prices.index, errors='coerce')
            
            # 유효한 날짜만 선택
            valid_dates = daily_avg_prices.index[~daily_avg_prices.index.isna()]
            
            if len(valid_dates) > 0:
                risk_timeline = pd.DataFrame(index=valid_dates)
                risk_timeline['risk_score'] = 0
                
                # 각 이벤트 날짜에 위험도 점수 부여
                if black_swan_df is not None and not black_swan_df.empty and 'date' in black_swan_df.columns:
                    for _, event in black_swan_df.iterrows():
                        try:
                            event_date = pd.to_datetime(event['date'], errors='coerce')
                            if pd.notnull(event_date) and event_date in risk_timeline.index:
                                risk_timeline.loc[event_date, 'risk_score'] += 3
                        except:
                            continue
                
                if extreme_events_df is not None and not extreme_events_df.empty and 'date' in extreme_events_df.columns:
                    for _, event in extreme_events_df.iterrows():
                        try:
                            event_date = pd.to_datetime(event['date'], errors='coerce')
                            if pd.notnull(event_date) and event_date in risk_timeline.index:
                                risk_timeline.loc[event_date, 'risk_score'] += 1
                        except:
                            continue
                
                # 30일 이동평균으로 위험도 추이 계산
                risk_timeline['smoothed_risk'] = risk_timeline['risk_score'].rolling(window=30, min_periods=1).mean()
                
                # 시각화
                fig, ax = plt.subplots(figsize=(10, 8))

                ax.plot(risk_timeline.index, risk_timeline['smoothed_risk'], label='Risk Level', color='red', linewidth=2)
                ax.fill_between(risk_timeline.index, 0, risk_timeline['smoothed_risk'], alpha=0.2, color='red')
                
                # 블랙 스완 이벤트 표시
                if black_swan_df is not None and not black_swan_df.empty and 'date' in black_swan_df.columns:
                    for _, event in black_swan_df.iterrows():
                        try:
                            event_date = pd.to_datetime(event['date'], errors='coerce')
                            if pd.notnull(event_date) and event_date in risk_timeline.index:
                                ax.scatter(event_date, risk_timeline.loc[event_date, 'smoothed_risk'],
                                          color='black', s=100, zorder=5, marker='*')
                        except:
                            continue
                
                ax.set_title('Risk Trend Over Time', fontsize=16)
                ax.set_xlabel('Date', fontsize=14)
                ax.set_ylabel('Risk Level', fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        except Exception as e:
            st.warning(f"위험도 추이 차트 생성 중 오류: {str(e)}")
    
    # Report download section
    st.markdown("### 📥 Download Report")

    report_text = f"""
    # 🧾 Executive Report on Market Extreme Events

    **Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}**

    ## 1. Market Risk Overview
    - Current Risk Level: {summary['risk_level']} (Score: {summary['risk_score']}/4)
    - Total Events Detected: {summary['total_events']}
    - Market Sentiment: {summary.get('current_sentiment', 'N/A')}

    ## 2. Category Analysis
    - Most Impacted Category: {summary['most_impacted']}
    - Most Stable Category: {summary['least_impacted']}

    ## 3. Event Breakdown
    - Black Swan Events: {summary['event_counts']['black_swan']}
    - Extreme Events: {summary['event_counts']['extreme']}
    - Technical Patterns: {summary['event_counts']['pattern']}
    - Correlation Breakdowns: {summary['event_counts']['correlation']}

    ## 4. Key Findings
    {chr(10).join(['- ' + finding for finding in summary['key_findings']])}
    
    ## 5. Investment Suggestions
    {chr(10).join(['- ' + suggestion for suggestion in summary['investment_suggestions']])}
    """

    st.download_button(
        label="📄 Download Report",
        data=report_text,
        file_name=f"extreme_events_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )
def extreme_events_dashboard(df, daily_avg_prices, daily_indicators, ml_predictions=None):
    """
    극단적 이벤트 대시보드를 생성합니다.
    
    Parameters:
    -----------
    df : DataFrame
        원본 데이터프레임
    daily_avg_prices : DataFrame
        일별 평균 가격 데이터
    daily_indicators : DataFrame
        일별 지표 데이터
    ml_predictions : dict, optional
        머신러닝 모드에서 생성된 예측 결과
    """
    st.header("🔍 Extreme Event Dashboard")
    
    # 머신러닝 예측 데이터 확인
    ml_predictions = ml_predictions or st.session_state.get('enhanced_model_data')
    
    if ml_predictions:
        st.success("✅ML predictions found. Including them in the analysis.")
    else:
        st.info("ℹ️ No ML predictions detected. Please run the ML mode for a full analysis.")
    
    # 대시보드 설명
    st.markdown("""
    Analyze black swan evets, market shocks, technical patterns, and more - all to power smarter investment decisions.
    """)
    #데이터 유효성 검사
    if df is None or df.empty:
        st.warning("No data for analysis")
        return
    if daily_avg_prices is None or daily_avg_prices.empty:
        st.warning("no daily avg prices data")
        return
    if daily_indicators is None or daily_indicators.empty:
        st.warning(" no daily indicator data")
        return
    
    #date type 
    try: 
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            st.success("Date column converted to datetime")
    except Exception as e:
        st.error(f"An error occurred during date conversion: {str(e)}")
        return
            
   # 감지 설정
    st.subheader("⚙️ Detection Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        black_swan_threshold = st.slider(
            "Black Swan Threshold(Z-score)",
            min_value=2.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            help="Extreme events are identified when values exceed 3.0 (99.7% confidence interval)"
        )
    
    with col2:
        extreme_percentile = st.slider(
            "Extreme economic indicator percentile",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="Top/Bottom n% events are flagged as extreme."
        )
    
    with col3:
        correlation_threshold = st.slider(
            "Correlation breakdown threshold",
            min_value=0.3,
            max_value=0.8,
            value=0.6,
            step=0.1,
            help="Sudden correlation shift threshold"
        )
    
    # 카테고리 선택
    if 'category' in df.columns:
        categories = df['category'].unique().tolist()
        selected_category = st.selectbox(
            "Select Category (Optional)",
            ["All"] + categories
        )
        
        category_filter = None if selected_category == "All" else selected_category
    else:
        category_filter = None
    
    # 감지 실행 버튼
    detect_events = st.button("🔍 Dectect Extreme Event")
    
    if detect_events:
        try:
            with st.spinner("Detecting special conditions and extreme events..."):
                # 1. 극단적 이벤트 감지 detect extreme event
                try:
                    extreme_events_df = detect_extreme_events(df, daily_avg_prices, daily_indicators)
                    if extreme_events_df is None:
                        extreme_events_df = pd.DataFrame(columns=['date', 'event_type', 'indicator', 'value', 'threshold', 'percentile', 'description'])
                        st.warning("No extreme events detected")
                except Exception as e:
                    st.error(f"error during extreme event detection: {str(e)}")
                    extreme_events_df = pd.DataFrame(columns=['date', 'event_type', 'indicator', 'value', 'threshold', 'percentile', 'description'])
                #2. Detect technical pattern 
                try:
                    technical_patterns_df = detect_technical_patterns(daily_avg_prices)
                    if technical_patterns_df is None:
                        technical_patterns_df = pd.DataFrame(columns=['date', 'category', 'pattern', 'description'])
                        st.warning("No extreme events detected")
                except Exception as e:
                    st.error(f"error during technical pattern detection: {str(e)}")
                    technical_patterns_df = pd.DataFrame(columns=['date', 'category', 'pattern', 'description'])
                #3 블랙 스완 이벤트 감지. Detect black swan event
                try:
                    black_swan_df = detect_black_swan_events(df, daily_avg_prices, daily_indicators, std_threshold=black_swan_threshold)
                    if black_swan_df is None:
                        black_swan_df = pd.DataFrame(columns=['date', 'category', 'event_type', 'z_score', 'return_pct', 'description'])
                        st.warning("No extreme events detected.")
                except Exception as e:
                    st.error(f"error during extreme event detection: {str(e)}")
                    black_swan_df = pd.DataFrame(columns=['date', 'category', 'event_type', 'z_score', 'return_pct', 'description'])

                # 4. 상관관계 붕괴 감지
                try:
                    correlation_breakdown_df = detect_correlations_breakdown(
                        daily_avg_prices,daily_indicators,
                        threshold= correlation_threshold
                    )
                    if correlation_breakdown_df is None:
                        correlation_breakdown_df= pd.DataFrame(columns=['date', 'pair', 'event_type', 'old_correlation', 'new_correlation', 'change', 'description'])
                        st.warning("상관관계 붕괴 이벤트를 감지할 수 없었습니다.")
                except Exception as e:
                    st.error(f"an error while detecting correlationg breake down: {str(e)}")
                    correlation_breakdown_df = pd.DataFrame(columns=['date', 'pair', 'event_type', 'old_correlation', 'new_correlation', 'change', 'description'])

                # 결과 저장 save results.
                st.session_state['extreme_events_df']= extreme_events_df
                st.session_state['black_swan_df'] = black_swan_df
                st.session_state['technical_patterns_df'] = technical_patterns_df
                st.session_state['correlation_breakdown_df'] = correlation_breakdown_df

                st.session_state['event_detected'] = True
                
            st.success("✅ Decected Extreme Event Analysis!")
        except Exception as e:
            st.error(f"이벤트 감지 중 오류가 발생했습니다: {str(e)}")
            return
        
        # 탭 생성
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Event Summary", 
            "💥 Black Swan", 
            "📈 Technical Patterns", 
            "📉 Extreme Economic Indicators",
            "🔄 Correlation Breakdown",
            "📄 Insight & Report"
        ])
        
        # 탭 1: 이벤트 요약
        with tab1:
            st.subheader("📊 Exceptional Market Events Summary")
            
            # 이벤트 개수 요약
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                black_swan_count = len(black_swan_df) if black_swan_df is not None and not black_swan_df.empty else 0
                st.metric("Black Swan", black_swan_count)
            
            with col2:
                pattern_count = len(technical_patterns_df) if technical_patterns_df is not None and not technical_patterns_df.empty else 0
                st.metric("Technical Patterns", pattern_count)
            
            with col3:
                extreme_econ_count = len(extreme_events_df) if extreme_events_df is not None and not extreme_events_df.empty else 0
                st.metric("Extreme Economic Indicators", extreme_econ_count)
            
            with col4:
                correlation_count = len(correlation_breakdown_df) if correlation_breakdown_df is not None and not correlation_breakdown_df.empty else 0
                st.metric("Correlation Breakdown", correlation_count)
            
            # # 카테고리별 이벤트 분포
            # if 'category' in df.columns:
            #     st.subheader("Category-wise Distribution of Extreme Events")
                
            #     # 각 이벤트 유형별 카테고리 분포 계산
            #     category_distributions = {}
                
            #     # 블랙 스완 분포
            #     if black_swan_df is not None and not black_swan_df.empty and 'category' in black_swan_df.columns:
            #         category_distributions['Black Swan'] = black_swan_df['category'].value_counts()
                
            #     # 기술적 패턴 분포
            #     if technical_patterns_df is not None and not technical_patterns_df.empty and 'category' in technical_patterns_df.columns:
            #         category_distributions['Technical Patterns'] = technical_patterns_df['category'].value_counts()
                
            #     # 결과 시각화
            #     if category_distributions:
            #         distribution_df = pd.DataFrame(category_distributions)
            #         # if event_dates and event_types:
            #         distribution_df = distribution_df.fillna(0)
                        
            #             # 막대 그래프 그리기
            #         fig, ax = plt.subplots(figsize=(7, 4))
            #         distribution_df.plot(kind='bar', ax=ax)

            #         plt.title('Category-wise Distribution of Extreme Events', fontsize=16)
            #         plt.xlabel('Category', fontsize=10)
            #         plt.ylabel('Event', fontsize=10)
            #         plt.xticks(rotation=45, ha='right')
            #         plt.legend(title='Event Type', fontsize= 8)
            #         plt.tight_layout()
            #         st.pyplot(fig)
            #         plt.close()
                             
        
            # 시간에 따른 이벤트 분포
            st.subheader("Distribution of Extreme Events Over Time")
            
            # 데이터 준비
            event_dates = []
            event_types = []
            
            # 블랙 스완 이벤트
            if black_swan_df is not None and not black_swan_df.empty:
                for _, row in black_swan_df.iterrows():
                    event_dates.append(row['date'])
                    event_types.append('Black Swan')
            
            # 기술적 패턴
            if technical_patterns_df is not None and not technical_patterns_df.empty:
                for _, row in technical_patterns_df.iterrows():
                    event_dates.append(row['date'])
                    event_types.append('Technical Patterns')
            
            # 극단적 경제 지표
            if extreme_events_df is not None and not extreme_events_df.empty:
                econ_extremes = extreme_events_df[
                    (extreme_events_df['event_type'] == 'Extreme High') | 
                    (extreme_events_df['event_type'] == 'Extreme Low')
                ]
                
                for _, row in econ_extremes.iterrows():
                    event_dates.append(row['date'])
                    event_types.append('Extreme Economic Indicators')
            
            # 상관관계 붕괴
            if correlation_breakdown_df is not None and not correlation_breakdown_df.empty:
                for _, row in correlation_breakdown_df.iterrows():
                    event_dates.append(row['date'])
                    event_types.append('Correlation Breakedown')
            
            # 이벤트 타임라인 생성
            if event_dates and event_types:
                events_df = pd.DataFrame({
                    'date': event_dates,
                    'event_type': event_types
                })
                try:
                    # 모든 날짜를 datetime으로 강제 변환
                    events_df['date'] = pd.to_datetime(events_df['date'], errors='coerce')

                    # 변환 실패한 날짜는 기본값으로 채우기
                    fallback_date = pd.Timestamp("1900-01-01")  # 또는 가장 오래된 날짜 이전
                    events_df['date'] = events_df['date'].fillna(fallback_date)

                    # 월별 이벤트 수 계산
                    events_df['yearmonth'] = events_df['date'].dt.strftime('%Y-%m')
                    monthly_events = events_df.groupby(['yearmonth', 'event_type']).size().unstack().fillna(0)

                    # 시각화
                    # 시각화
                    fig, ax = plt.subplots(figsize=(12, 6))  # 가로를 넓게
                    monthly_events.plot(kind='bar', stacked=True, ax=ax)

                    # 제목, 축 레이블
                    plt.title('Monthly Frequency of Extreme Events', fontsize=12)
                    plt.xlabel('Year-Month', fontsize=10)
                    plt.ylabel('Number of Events', fontsize=10)

                    # x축 레이블 설정
                    plt.xticks(rotation=45, ha='right', fontsize=8)

                    # 범례
                    plt.legend(title='Event Type', fontsize=8)

                    # y축 범위 제한해서 너무 긴 바 방지
                    ax.set_ylim(0, 25)   # ✅ 필요에 맞게 조정해

                    # 레이아웃 정리
                    plt.tight_layout()

                    # Streamlit 출력
                    st.pyplot(fig)
                    plt.close()


                except Exception as e:
                    st.error(f"📛 이벤트 타임라인 처리 중 오류 발생: {str(e)}")
                # 1. Korrelationseinbruch zwischen Verteidigungsindustrie und CPI (erstes Bild)

                st.markdown("""
                            # 1. Korrelationseinbruch zwischen Verteidigungsindustrie und CPI (erstes Bild)

                            ## Kernaussagen:
                            - Am 10. Mai 2024 stieg die Korrelation von 0,04 auf 0,66 (+0,62).
                            - Vorher: kaum Zusammenhang → Nachher: starke positive Korrelation.
                            - Strukturwandel: Inflation beeinflusst zunehmend Investitionen in die Verteidigungsindustrie.

                            ## Implikationen:
                            - Inflation wird mit geopolitischen Spannungen verknüpft.
                            - CPI-Anstieg wird als Signal für steigende Aktienkurse der Verteidigungsbranche neu bewertet.

                            """)
        
        
            # 최근 극단 이벤트 목록
            st.subheader("Recently Detected Extreme Events")
            
            # 모든 이벤트 통합
            all_events = []
            
            # 블랙 스완 이벤트
            if black_swan_df is not None and not black_swan_df.empty:
                for _, row in black_swan_df.iterrows():
                    all_events.append({
                        'date': row['date'],
                        'event_type': 'Black swan',
                        'description': row['description'],
                        'category': row['category'] if 'category' in row else None
                    })
            
            # 기술적 패턴
            if technical_patterns_df is not None and not technical_patterns_df.empty:
                for _, row in technical_patterns_df.iterrows():
                    all_events.append({
                        'date': row['date'],
                        'event_type': 'Technical Patterns',
                        'description': f"{row['pattern']}: {row['description']}",
                        'category': row['category'] if 'category' in row else None
                    })
            
            # 극단적 경제 지표
            if extreme_events_df is not None and not extreme_events_df.empty:
                econ_extremes = extreme_events_df[
                    (extreme_events_df['event_type'] == 'Extreme High') | 
                    (extreme_events_df['event_type'] == 'Extreme Low')
                ]
                
                for _, row in econ_extremes.iterrows():
                    all_events.append({
                        'date': row['date'],
                        'event_type': 'Extreme Economic Indicators',
                        'description': row['description'],
                        'category': None
                    })
            
            # 상관관계 붕괴
            if correlation_breakdown_df is not None and not correlation_breakdown_df.empty:
                for _, row in correlation_breakdown_df.iterrows():
                    all_events.append({
                        'date': row['date'],
                        'event_type': 'Correlation Breakdown',
                        'description': row['description'],
                        'category': row['pair'] if 'pair' in row else None
                    })
            
            # 모든 이벤트 표시
            if all_events:
                events_df = pd.DataFrame(all_events)
                events_df = events_df.sort_values('date', ascending=False)
                
                # 카테고리 필터링 (필요시)
                if category_filter:
                    events_df = events_df[events_df['category'] == category_filter]
                
                # 최근 20개 이벤트만 표시
                recent_events = events_df.head(20)
                
                st.dataframe(
                    recent_events[['date', 'event_type', 'description', 'category']],
                    column_config={
                        "date": st.column_config.DatetimeColumn("date"),
                        "event_type": st.column_config.TextColumn("event_type"),
                        "description": st.column_config.TextColumn("description"),
                        "category": st.column_config.TextColumn("category")
                    }
                )
            else:
                st.info("감지된 극단 이벤트가 없습니다.")
        
        # 탭 2: 블랙 스완 이벤트
        with tab2:
            st.subheader("💥 Black Swan Events")
            
            if black_swan_df is not None and not black_swan_df.empty:
                # 필터링된 데이터
                filtered_bs = black_swan_df
                if category_filter:
                    filtered_bs = black_swan_df[black_swan_df['category'] == category_filter]
                
                if not filtered_bs.empty:
                    # 블랙 스완 이벤트 표시
                    st.write("detected Black Swan Event:")
                    
                    st.dataframe(
                        filtered_bs.sort_values('date', ascending=False),
                        column_config={
                            "date": st.column_config.DatetimeColumn("date"),
                            "category": st.column_config.TextColumn("category"),
                            "event_type": st.column_config.TextColumn("event_type"),
                            "z_score": st.column_config.NumberColumn("Z-score", format="%.2f"),
                            "return_pct": st.column_config.NumberColumn("return_pct (%)", format="%.2f"),
                            "description": st.column_config.TextColumn("description")
                        }
                    )
                    
                    # 블랙 스완 시각화
                    try: 
                        visualize_extreme_events(None, filtered_bs, daily_avg_prices, daily_indicators, st.session_state.get('technical_patterns_df'), category_filter)
                    except Exception as e:
                        st.error(f"😭error occured while visualizing black swan event {str(e)}")
                    # 블랙 스완 이벤트의 수익률 영향 분석
                    try: 
                        analyze_extreme_events_impact(df, None, filtered_bs, category_filter)
                    except Exception as e:
                        st.error(f"😭error occurred while analyzing the impact of black swan event: {str(e)}")
                else:
                    st.info(f"선택한 카테고리 '{category_filter}'에 대한 블랙 스완 이벤트가 없습니다.")
            else:
                st.info("감지된 블랙 스완 이벤트가 없습니다.")
        
        # 탭 3: 기술적 패턴
        with tab3:
            st.subheader("📈 Technical Patterns")
            
            if technical_patterns_df is not None and not technical_patterns_df.empty:
                # 필터링된 데이터
                filtered_patterns = technical_patterns_df
                if category_filter:
                    filtered_patterns = technical_patterns_df[technical_patterns_df['category'] == category_filter]
                
                if not filtered_patterns.empty:
                    # 기술적 패턴 표시
                    st.write("Detected Technical Patterns:")
                    
                    st.dataframe(
                        filtered_patterns.sort_values('date', ascending=False),
                        column_config={
                            "date": st.column_config.DatetimeColumn("date"),
                            "category": st.column_config.TextColumn("category"),
                            "pattern": st.column_config.TextColumn("pattern"),
                            "description": st.column_config.TextColumn("description")
                        }
                    )
                    
                    # 패턴 유형별 분포 시각화와 함께 표시할 다른 내용 준비
                    col1, col2 = st.columns(2)
                    
                    # 첫 번째 컬럼: 패턴 유형별 분포
                    with col1:
                        try: 
                            if 'pattern' not in filtered_patterns.columns or filtered_patterns['pattern'].empty:
                                st.info("pattern data is missing or empty")
                            else:
                                pattern_counts = filtered_patterns['pattern'].value_counts()

                            if pattern_counts.empty:
                                st.info("impossible to calculate Pattern distribution😭")
                            else:
                                fig, ax = plt.subplots(figsize=(8, 6))
                                pattern_counts.plot(kind='bar', ax=ax, color='teal')
                                
                                plt.title('Frequency of Technical Pattern by Type', fontsize=16)
                                plt.xlabel('pattern', fontsize=14)
                                plt.ylabel('frequency', fontsize=14)
                                plt.xticks(rotation=45, ha='right')
                                plt.tight_layout()
                                
                                st.pyplot(fig)
                                plt.close()
                        except Exception as e:
                            st.error(f"error occurred while visualization pattern distribution")
                    
                    # 두 번째 컬럼: 카테고리별 패턴 분포
                    with col2:
                        try:
                            if 'category' in filtered_patterns.columns and 'pattern' in filtered_patterns.columns:
                                # 카테고리별 패턴 분포 계산
                                category_pattern_counts = filtered_patterns.groupby(['category', 'pattern']).size().unstack().fillna(0)
                                
                                if not category_pattern_counts.empty and category_pattern_counts.shape[0] > 0 and category_pattern_counts.shape[1] > 0:
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    category_pattern_counts.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
                                    
                                    plt.title('Patterns Distribution by Category', fontsize=16)
                                    plt.xlabel('Category', fontsize=14)
                                    plt.ylabel('Count', fontsize=14)
                                    plt.xticks(rotation=45, ha='right')
                                    plt.legend(title='Pattern', bbox_to_anchor=(1.05, 1), loc='upper left')
                                    plt.tight_layout()
                                    
                                    st.pyplot(fig)
                                    plt.close()
                                else:
                                    st.info("Not enough data to visualize patterns by category")
                            else:
                                st.info("Required columns 'category' or 'pattern' not found")
                        except Exception as e:
                            st.error(f"Error analyzing patterns by category: {str(e)}")
                    
                    # 대표적인 패턴에 대한 가격 차트 시각화
                    if daily_avg_prices is not None and not daily_avg_prices.empty:
                        st.subheader("Main Technical Pattern Visualization")
                        
                        try:
                            # 가장 많이 발생한 패턴 선택
                            if 'pattern' in filtered_patterns.columns:
                                pattern_counts = filtered_patterns['pattern'].value_counts()
                                if not pattern_counts.empty:
                                    top_patterns = pattern_counts.head(min(6, len(pattern_counts))).index.tolist()
                                    
                                    # 패턴 데이터 처리 및 시각화
                                    pattern_examples = []
                                    
                                    # 모든 패턴 예시 데이터 수집
                                    for pattern in top_patterns:
                                        pattern_data = filtered_patterns[filtered_patterns['pattern'] == pattern]
                                        
                                        if not pattern_data.empty:
                                            # 각 패턴에 대해 최대 2개 예시 가져오기
                                            for _, row in pattern_data.head(2).iterrows():
                                                pattern_examples.append({
                                                    'pattern': pattern,
                                                    'date': row['date'],
                                                    'category': row['category'],
                                                    'description': row.get('description', '')
                                                })
                                    
                                    # 예시가 짝수가 아니면 하나를 제거하여 짝수로 만들기
                                    if len(pattern_examples) % 2 != 0 and len(pattern_examples) > 0:
                                        pattern_examples = pattern_examples[:-1]
                                    
                                    # 2개씩 나눠서 그리드 레이아웃으로 표시
                                    for i in range(0, len(pattern_examples), 2):
                                        cols = st.columns(2)
                                        
                                        for j in range(2):
                                            if i + j < len(pattern_examples):
                                                example = pattern_examples[i + j]
                                                
                                                with cols[j]:
                                                    pattern_date = example['date']
                                                    pattern_name = example['pattern']
                                                    pattern_category = example['category']
                                                    pattern_desc = example['description']
                                                    
                                                    # 패턴 설명 표시
                                                    st.markdown(f"**{pattern_name}** - {pattern_date.strftime('%Y-%m-%d')}")
                                                    if pattern_desc:
                                                        st.markdown(f"*{pattern_desc}*", help=pattern_desc)
                                                    
                                                    # 패턴 전후 30일 기간
                                                    start_date = pattern_date - pd.Timedelta(days=30)
                                                    end_date = pattern_date + pd.Timedelta(days=30)
                                                    
                                                    # 데이터 필터링
                                                    mask = (daily_avg_prices.index >= start_date) & (daily_avg_prices.index <= end_date)
                                                    period_data = daily_avg_prices.loc[mask]
                                                    
                                                    if not period_data.empty and pattern_category in period_data.columns:
                                                        # 이동평균선 계산
                                                        ma5 = period_data[pattern_category].rolling(window=5).mean()
                                                        ma20 = period_data[pattern_category].rolling(window=20).mean()
                                                        ma50 = period_data[pattern_category].rolling(window=50).mean()
                                                        
                                                        # 차트 그리기
                                                        fig, ax = plt.subplots(figsize=(8, 6))
                                                        
                                                        ax.plot(period_data.index, period_data[pattern_category], label=pattern_category, linewidth=2)
                                                        ax.plot(period_data.index, ma5, label='5-Day MA', linestyle='--')
                                                        ax.plot(period_data.index, ma20, label='20-Day MA', linestyle='--')
                                                        ax.plot(period_data.index, ma50, label='50-Day MA', linestyle='--')
                                                        
                                                        # 패턴 발생일 표시
                                                        if pattern_date in period_data.index:
                                                            pattern_price = period_data.loc[pattern_date, pattern_category]
                                                            ax.scatter(pattern_date, pattern_price, color='red', s=100, zorder=5)
                                                            ax.annotate(
                                                                pattern_name,
                                                                (pattern_date, pattern_price),
                                                                textcoords="offset points",
                                                                xytext=(0, 10),
                                                                ha='center',
                                                                fontweight='bold',
                                                                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7)
                                                            )
                                                        
                                                        ax.set_title(f"{pattern_category} - {pattern_name}", fontsize=14)
                                                        ax.set_xlabel('Date', fontsize=12)
                                                        ax.set_ylabel('Price', fontsize=12)
                                                        ax.legend()
                                                        ax.grid(True, alpha=0.3)
                                                        
                                                        plt.xticks(rotation=45)
                                                        plt.tight_layout()
                                                        
                                                        st.pyplot(fig)
                                                        plt.close()
                                                    else:
                                                        st.warning(f"No data available for {pattern_category} in the selected period")
                                else:
                                    st.warning("No patterns found to visualize")
                            else:
                                st.warning("'pattern' column not found in the data")
                            
                            # 추가 시각화: 패턴 발생 시간대별 분포
                            st.subheader("Pattern Time Distribution Analysis")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                try:
                                    if 'date' in filtered_patterns.columns:
                                        # 년/월별 패턴 발생 빈도
                                        filtered_patterns['year_month'] = filtered_patterns['date'].dt.strftime('%Y-%m')
                                        monthly_counts = filtered_patterns.groupby('year_month').size()
                                        
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        monthly_counts.plot(kind='line', marker='o', ax=ax, color='purple')
                                        
                                        plt.title('Monthly Pattern Occurrence', fontsize=14)
                                        plt.xlabel('Year-Month', fontsize=12)
                                        plt.ylabel('Number of Patterns', fontsize=12)
                                        plt.xticks(rotation=45)
                                        plt.grid(True, alpha=0.3)
                                        plt.tight_layout()
                                        
                                        st.pyplot(fig)
                                        plt.close()
                                    else:
                                        st.info("Date information not available")
                                except Exception as e:
                                    st.error(f"Error analyzing time distribution: {str(e)}")
                            
                            with col2:
                                try:
                                    if 'date' in filtered_patterns.columns and 'pattern' in filtered_patterns.columns:
                                        # 최근 1년 데이터 필터링
                                        last_date = filtered_patterns['date'].max()
                                        one_year_ago = last_date - pd.Timedelta(days=365)
                                        recent_patterns = filtered_patterns[filtered_patterns['date'] >= one_year_ago]
                                        
                                        if not recent_patterns.empty:
                                            # 패턴별 월간 분포 (히트맵)
                                            recent_patterns['month'] = recent_patterns['date'].dt.month
                                            pattern_month_counts = recent_patterns.groupby(['pattern', 'month']).size().unstack().fillna(0)
                                            
                                            fig, ax = plt.subplots(figsize=(8, 6))
                                            sns.heatmap(pattern_month_counts, cmap='YlGnBu', annot=True, fmt='g', ax=ax)
                                            
                                            plt.title('Monthly Pattern Distribution (Last Year)', fontsize=14)
                                            plt.xlabel('Month', fontsize=12)
                                            plt.ylabel('Pattern', fontsize=12)
                                            plt.tight_layout()
                                            
                                            st.pyplot(fig)
                                            plt.close()
                                        else:
                                            st.info("Not enough recent data for monthly pattern analysis")
                                    else:
                                        st.info("Required columns not available")
                                except Exception as e:
                                    st.error(f"Error creating pattern heatmap: {str(e)}")
                            
                        except Exception as e:
                            st.error(f"Error creating technical pattern visualization: {str(e)}")
        
        # 탭 4: 경제 지표 극단값
        # 탭 4: 경제 지표 극단값
        with tab4:
            st.subheader("📉 Extreme Economic Indicators")
            
            if extreme_events_df is not None and not extreme_events_df.empty:
                # 경제 지표 극단값 필터링
                economic_extremes = extreme_events_df[
                    ((extreme_events_df['event_type'] == 'Extreme High') | 
                    (extreme_events_df['event_type'] == 'Extreme Low')) &
                    (extreme_events_df['indicator'].str.contains('_norm'))
                ]
                
                if not economic_extremes.empty:
                    # 경제 지표 극단값 표시
                    st.write("Detected Extreme Economic Indicators:")
                    
                    st.dataframe(
                        economic_extremes.sort_values('date', ascending=False),
                        column_config={
                            "date": st.column_config.DatetimeColumn("date"),
                            "event_type": st.column_config.TextColumn("event_type"),
                            "indicator": st.column_config.TextColumn("indicator"),
                            "value": st.column_config.NumberColumn("value", format="%.2f"),
                            "threshold": st.column_config.NumberColumn("threshold", format="%.2f"),
                            "percentile": st.column_config.NumberColumn("percentile"),
                            "description": st.column_config.TextColumn("description")
                        }
                    )
                    
                    # 경제 지표 극단값 시각화 개선 - 직접 구현
                    st.subheader("Extreme Economic Indicators Visualization")
                    
                    # 상위 지표 선택
                    top_indicators = economic_extremes['indicator'].value_counts().head(6).index.tolist()
                    num_indicators = len(top_indicators)
                    
                    # 행과 열 계산 (2개의 열)
                    num_rows = (num_indicators + 1) // 2
                    
                    for row in range(num_rows):
                        cols = st.columns(2)
                        
                        for col_idx in range(2):
                            indicator_idx = row * 2 + col_idx
                            
                            if indicator_idx < num_indicators:
                                indicator = top_indicators[indicator_idx]
                                clean_indicator = indicator.replace('_norm', '')
                                
                                with cols[col_idx]:
                                    try:
                                        # 지표에 해당하는 극단 이벤트 추출
                                        indicator_events = economic_extremes[economic_extremes['indicator'] == indicator]
                                        
                                        if not indicator_events.empty and daily_indicators is not None:
                                            # 원본 지표 데이터 추출
                                            indicator_data = daily_indicators[indicator] if indicator in daily_indicators.columns else None
                                            
                                            if indicator_data is not None:
                                                fig, ax = plt.subplots(figsize=(8, 6))
                                                
                                                # 지표 데이터 플롯
                                                ax.plot(daily_indicators.index, indicator_data, label=clean_indicator, color='blue')
                                                
                                                # 극단적 이벤트 표시
                                                for _, event in indicator_events.iterrows():
                                                    event_date = event['date']
                                                    if event_date in daily_indicators.index:
                                                        value_at_event = indicator_data.loc[event_date]
                                                        event_type = event['event_type']
                                                        color = 'red' if event_type == 'Extreme High' else 'green'
                                                        
                                                        ax.scatter(event_date, value_at_event, color=color, s=80, zorder=5)
                                                        
                                                
                                                # 90% 및 10% 분위수 라인 추가
                                                high_threshold = indicator_data.quantile(0.90)
                                                low_threshold = indicator_data.quantile(0.10)
                                                
                                                ax.axhline(y=high_threshold, color='red', linestyle='--', 
                                                        label=f'90% threshold ({high_threshold:.2f})')
                                                ax.axhline(y=low_threshold, color='green', linestyle='--', 
                                                        label=f'10% threshold ({low_threshold:.2f})')
                                                
                                                ax.set_title(f'{clean_indicator} Extreme Events', fontsize=14)
                                                ax.set_xlabel('Date', fontsize=12)
                                                ax.set_ylabel(clean_indicator, fontsize=12)
                                                ax.legend()
                                                ax.grid(True, alpha=0.3)
                                                
                                                plt.xticks(rotation=45)
                                                plt.tight_layout()
                                                
                                                st.pyplot(fig)
                                                plt.close()
                                            else:
                                                st.warning(f"No data found for indicator: {indicator}")
                                        else:
                                            st.info(f"No extreme events for {clean_indicator}")
                                    except Exception as e:
                                        st.error(f"Error visualizing {clean_indicator}: {str(e)}")
                    
                    # 영향 분석 섹션 - 두 컬럼으로 표시
                    st.subheader("Impact Analysis of Extreme Economic Indicators")
                    
                    # 경제 지표 유형별 영향 분석
                    if economic_extremes is not None and not economic_extremes.empty and df is not None:
                        try:
                            # 주요 경제 지표 그룹 선택
                            indicator_groups = []
                            for ind in economic_extremes['indicator'].unique():
                                base_name = ind.split('_')[0]  # '_norm' 제거
                                if base_name not in indicator_groups:
                                    indicator_groups.append(base_name)
                            
                            # 최대 4개 그룹만 선택
                            indicator_groups = indicator_groups[:min(4, len(indicator_groups))]
                            num_groups = len(indicator_groups)
                            rows_needed = (num_groups + 1) // 2
                            
                            for row in range(rows_needed):
                                cols = st.columns(2)
                                
                                for col_idx in range(2):
                                    group_idx = row * 2 + col_idx
                                    
                                    if group_idx < num_groups:
                                        base_ind = indicator_groups[group_idx]
                                        
                                        with cols[col_idx]:
                                            # 해당 경제 지표에 관련된 모든 극단 이벤트 필터링
                                            related_events = economic_extremes[economic_extremes['indicator'].str.contains(base_ind)]
                                            
                                            if not related_events.empty:
                                                # 이벤트 유형별 집계
                                                event_types = related_events['event_type'].value_counts()
                                                
                                                # 바 차트로 시각화
                                                fig, ax = plt.subplots(figsize=(8, 6))
                                                event_types.plot(kind='bar', color=['red', 'green'], ax=ax)
                                                
                                                plt.title(f'{base_ind} Extreme Event Types', fontsize=14)
                                                plt.xlabel('Event Type', fontsize=12)
                                                plt.ylabel('Count', fontsize=12)
                                                plt.xticks(rotation=0)
                                                plt.tight_layout()
                                                
                                                st.pyplot(fig)
                                                plt.close()
                                                
                                                # 이벤트 전후 수익률 분석
                                                if 'date' in related_events.columns and category_filter:
                                                    dates = related_events['date'].tolist()
                                                    
                                                    # 각 이벤트 날짜 전후 수익률 계산
                                                    returns_before = []
                                                    returns_after = []
                                                    
                                                    for event_date in dates:
                                                        # 이벤트 전 5일 수익률
                                                        start_before = event_date - pd.Timedelta(days=7)
                                                        price_data = df[(df['date'] >= start_before) & 
                                                                    (df['date'] <= event_date) & 
                                                                    (df['category'] == category_filter)]
                                                        
                                                        if len(price_data) >= 2:
                                                            start_price = price_data.iloc[0]['close']
                                                            end_price = price_data.iloc[-1]['close']
                                                            returns_before.append((end_price / start_price - 1) * 100)
                                                        
                                                        # 이벤트 후 5일 수익률
                                                        end_after = event_date + pd.Timedelta(days=7)
                                                        price_data = df[(df['date'] >= event_date) & 
                                                                    (df['date'] <= end_after) & 
                                                                    (df['category'] == category_filter)]
                                                        
                                                        if len(price_data) >= 2:
                                                            start_price = price_data.iloc[0]['close']
                                                            end_price = price_data.iloc[-1]['close']
                                                            returns_after.append((end_price / start_price - 1) * 100)
                                                    
                                                    # 결과 시각화
                                                    if returns_before and returns_after:
                                                        fig, ax = plt.subplots(figsize=(8, 6))
                                                        
                                                        # 박스플롯으로 비교
                                                        box_data = [returns_before, returns_after]
                                                        ax.boxplot(box_data, labels=['Before Event', 'After Event'])
                                                        
                                                        # 포인트 추가
                                                        for i, data in enumerate([returns_before, returns_after], 1):
                                                            x = np.random.normal(i, 0.04, size=len(data))
                                                            ax.scatter(x, data, alpha=0.6)
                                                        
                                                        plt.title(f'{base_ind} Impact on {category_filter} Returns', fontsize=14)
                                                        plt.ylabel('Return (%)', fontsize=12)
                                                        plt.grid(True, alpha=0.3)
                                                        plt.tight_layout()
                                                        
                                                        st.pyplot(fig)
                                                        plt.close()
                                                    else:
                                                        st.info(f"Not enough price data to analyze returns for {base_ind}")
                                            else:
                                                st.info(f"No extreme events found for {base_ind}")
                        except Exception as e:
                            st.error(f"Error analyzing indicator impact: {str(e)}")
                    
                    # 기존 함수 호출 (백업)
                    # visualize_extreme_events(economic_extremes, None, daily_avg_prices, daily_indicators, category_filter)
                    # analyze_extreme_events_impact(df, economic_extremes, None, category_filter)
                else:
                    st.info("감지된 경제 지표 극단값이 없습니다.")
            else:
                st.info("감지된 경제 지표 극단값이 없습니다.")

        # 탭 5: 상관관계 붕괴
        with tab5:
            st.subheader("🔄 Correlation Breakdown")
            
            # 디버깅 정보를 숨기거나 확장 가능한 섹션으로 정리
            with st.expander("Debugging information"):
                st.write(f"correlation_breakdown_df exists: {correlation_breakdown_df is not None}")
                st.write(f"correlation_breakdown_df empty: {correlation_breakdown_df.empty if correlation_breakdown_df is not None else 'N/A'}")
                st.write(f"daily_avg_prices exists: {daily_avg_prices is not None}")
                st.write(f"daily_avg_prices empty: {daily_avg_prices.empty if daily_avg_prices is not None else 'N/A'}")
            
            if correlation_breakdown_df is not None and not correlation_breakdown_df.empty:
                # 필터링된 데이터
                filtered_corr = correlation_breakdown_df
                if category_filter:
                    filtered_corr = correlation_breakdown_df[correlation_breakdown_df['pair'].str.contains(category_filter)]
                
                # 필터링 후 데이터 상태 확인 (확장 가능한 섹션으로 이동)
                with st.expander("Filtered data information"):
                    st.write(f"filtered_corr empty: {filtered_corr.empty}")
                
                if not filtered_corr.empty:
                    # 상관관계 붕괴 이벤트 표시
                    st.write("Detected Correlation Breakdown:")
                    
                    st.dataframe(
                        filtered_corr.sort_values('date', ascending=False),
                        column_config={
                            "date": st.column_config.DatetimeColumn("date"),
                            "pair": st.column_config.TextColumn("pair"),
                            "event_type": st.column_config.TextColumn("event_type"),
                            "old_correlation": st.column_config.NumberColumn("old_correlation", format="%.2f"),
                            "new_correlation": st.column_config.NumberColumn("new_correlation", format="%.2f"),
                            "change": st.column_config.NumberColumn("change", format="%.2f"),
                            "description": st.column_config.TextColumn("description")
                        }
                    )
                    
                    # 상관관계 붕괴 시각화
                    if daily_avg_prices is not None and not daily_avg_prices.empty:
                        st.subheader("Main Correlation Breakdown Visualization")
                        
                        if 'change' in filtered_corr.columns:
                            # 변경된 부분: 상위 4개로 늘리고 그리드 레이아웃 적용
                            top_changes = filtered_corr.sort_values('change', key=abs, ascending=False).head(4)
                            
                            # 2개씩 나누어 표시
                            for i in range(0, len(top_changes), 2):
                                cols = st.columns(2)
                                
                                for j in range(2):
                                    if i + j < len(top_changes):
                                        row = top_changes.iloc[i + j]
                                        
                                        with cols[j]:
                                            pair = row['pair'].split('-')
                                            
                                            if len(pair) == 2:
                                                item1, item2 = pair
                                                date = row['date']
                                                
                                                # 데이터 준비 - 개선된 로직
                                                series1 = None
                                                series2 = None
                                                
                                                # item1 검색
                                                if item1 in daily_avg_prices.columns:
                                                    series1 = daily_avg_prices[item1]
                                                elif daily_indicators is not None:
                                                    if item1 in daily_indicators.columns:
                                                        series1 = daily_indicators[item1]
                                                    else:
                                                        # 부분 일치 검색 (GDP_norm -> GDP)
                                                        for col in daily_indicators.columns:
                                                            if item1 in col:
                                                                series1 = daily_indicators[col]
                                                                break
                                                
                                                # item2 검색
                                                if item2 in daily_avg_prices.columns:
                                                    series2 = daily_avg_prices[item2]
                                                elif daily_indicators is not None:
                                                    if item2 in daily_indicators.columns:
                                                        series2 = daily_indicators[item2]
                                                    else:
                                                        # 부분 일치 검색
                                                        for col in daily_indicators.columns:
                                                            if item2 in col:
                                                                series2 = daily_indicators[col]
                                                                break
                                                
                                                # 데이터가 있는 경우에만 시각화 진행
                                                if series1 is not None and series2 is not None:
                                                    # 상관관계 전후 60일 기간
                                                    start_date = date - pd.Timedelta(days=60)
                                                    end_date = date + pd.Timedelta(days=60)
                                                    
                                                    # 데이터 필터링
                                                    mask1 = (series1.index >= start_date) & (series1.index <= end_date)
                                                    mask2 = (series2.index >= start_date) & (series2.index <= end_date)
                                                    
                                                    period_data1 = series1.loc[mask1]
                                                    period_data2 = series2.loc[mask2]
                                                    
                                                    # 공통 날짜 찾기
                                                    common_dates = period_data1.index.intersection(period_data2.index)
                                                    
                                                    if len(common_dates) > 10:  # 충분한 데이터가 있는 경우
                                                        # 시계열 표준화 (비교 가능하게)
                                                        normalized1 = (period_data1[common_dates] - period_data1[common_dates].mean()) / period_data1[common_dates].std()
                                                        normalized2 = (period_data2[common_dates] - period_data2[common_dates].mean()) / period_data2[common_dates].std()
                                                        
                                                        # 차트 그리기 - 그래프 크기 조정
                                                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                                                        
                                                        # 시계열 차트
                                                        ax1.plot(common_dates, normalized1, label=item1)
                                                        ax1.plot(common_dates, normalized2, label=item2)
                                                        
                                                        # 상관관계 붕괴 날짜 표시
                                                        ax1.axvline(x=date, color='red', linestyle='--', label='Correlation Breakdown')
                                                        
                                                        # 타이틀 수정
                                                        ax1.set_title(f'{item1} vs {item2}', fontsize=14)
                                                        ax1.set_ylabel('Normalized Value', fontsize=12)
                                                        ax1.legend()
                                                        ax1.grid(True, alpha=0.3)
                                                        
                                                        # 롤링 상관계수
                                                        combined = pd.DataFrame({
                                                            item1: period_data1[common_dates],
                                                            item2: period_data2[common_dates]
                                                        })
                                                        
                                                        rolling_corr = combined[item1].rolling(window=30).corr(combined[item2])
                                                        rolling_corr = rolling_corr.fillna(0)
                                                        
                                                        ax2.plot(common_dates, rolling_corr, color='purple', linewidth=2)
                                                        ax2.axvline(x=date, color='red', linestyle='--')
                                                        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                                                        
                                                        # 이전 및 새 상관계수 표시
                                                        ax2.axhline(y=row['old_correlation'], color='green', linestyle=':', 
                                                                    label=f'Old: {row["old_correlation"]:.2f}')
                                                        ax2.axhline(y=row['new_correlation'], color='blue', linestyle=':', 
                                                                    label=f'New: {row["new_correlation"]:.2f}')
                                                        
                                                        ax2.set_xlabel('Date', fontsize=12)
                                                        ax2.set_ylabel('30-Day Rolling Correlation', fontsize=12)
                                                        ax2.set_ylim(-1.1, 1.1)
                                                        ax2.legend()
                                                        ax2.grid(True, alpha=0.3)
                                                        
                                                        plt.xticks(rotation=45)
                                                        plt.tight_layout()
                                                        
                                                        st.pyplot(fig)
                                                        plt.close()
                                                    else:
                                                        st.warning(f"Not enough common data points for {item1} and {item2}")
                                                else:
                                                    st.warning(f"Data not found for {item1} or {item2}")
                                            else:
                                                st.warning(f"Invalid pair format: {row['pair']}")
                            
                            # 상관관계 변화 분석 추가 섹션
                            st.subheader("Correlation Change Analysis")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                try:
                                    # 상관관계 변화 분포 시각화
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    sns.histplot(filtered_corr['change'], bins=10, kde=True, ax=ax)
                                    
                                    plt.title('Distribution of Correlation Changes', fontsize=14)
                                    plt.xlabel('Correlation Change', fontsize=12)
                                    plt.ylabel('Frequency', fontsize=12)
                                    plt.axvline(x=0, color='red', linestyle='--')
                                    plt.grid(True, alpha=0.3)
                                    plt.tight_layout()
                                    
                                    st.pyplot(fig)
                                    plt.close()
                                except Exception as e:
                                    st.error(f"Error creating correlation change distribution: {str(e)}")
                            
                            with col2:
                                try:
                                    # 상관관계 변화 시간 추세
                                    if 'date' in filtered_corr.columns:
                                        filtered_corr = filtered_corr.sort_values('date')
                                        
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        ax.scatter(filtered_corr['date'], filtered_corr['change'], 
                                                c=filtered_corr['change'].apply(lambda x: 'red' if x < 0 else 'green'),
                                                alpha=0.7, s=50)
                                        
                                        # 추세선 추가
                                        z = np.polyfit(np.arange(len(filtered_corr)), filtered_corr['change'], 1)
                                        p = np.poly1d(z)
                                        ax.plot(filtered_corr['date'], p(np.arange(len(filtered_corr))), 
                                            "r--", linewidth=2, label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
                                        
                                        plt.title('Correlation Changes Over Time', fontsize=14)
                                        plt.xlabel('Date', fontsize=12)
                                        plt.ylabel('Correlation Change', fontsize=12)
                                        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                                        plt.grid(True, alpha=0.3)
                                        plt.legend()
                                        plt.xticks(rotation=45)
                                        plt.tight_layout()
                                        
                                        st.pyplot(fig)
                                        plt.close()
                                    else:
                                        st.info("Date information not available for trend analysis")
                                except Exception as e:
                                    st.error(f"Error creating correlation trend visualization: {str(e)}")
                        else:
                            st.warning("'change' column not found in filtered_corr")
                    else:
                        st.warning("No daily average price data available for correlation breakdown visualization")
                else:
                    st.info(f"No correlation breakdown events found for the selected category '{category_filter}'")

        with tab6:
            st.subheader("📄 Insight & Executive Report")
            
            # 세션에서 머신러닝 예측 결과 가져오기
            ml_predictions = st.session_state.get('enhanced_model_data')
            
            # 디버깅 정보
            #st.write(f"DEBUG: 머신러닝 예측 결과 존재 여부: {ml_predictions is not None}")
            #if ml_predictions is not None:
                #st.write(f"DEBUG: 예측 결과 카테고리: {list(ml_predictions.keys())}")
            
            # 현재 모드에서 사용할 기본 model_data 초기화
            # 기본 model_data 생성 (이미 daily_avg_prices가 있다고 가정)
            model_data = {}
            for category in daily_avg_prices.columns:
                model_data[category] = pd.DataFrame(index=daily_avg_prices.index)
                model_data[category]['close'] = daily_avg_prices[category]


            # extreme_enhanced_model_data 생성
            if 'extreme_events_df' in st.session_state and 'black_swan_df' in st.session_state:
                try:
                    with st.spinner("Creating enhanced data with extreme events..."):
                        # create_extreme_enhanced_data 함수 호출
                        extreme_enhanced_model_data = create_extreme_enhanced_data(
                            model_data=model_data,  # 초기화한 기본 모델 데이터
                            extreme_events_df=st.session_state['extreme_events_df'],
                            black_swan_df=st.session_state['black_swan_df'],
                            technical_patterns_df=st.session_state.get('technical_patterns_df'),
                            correlation_breakdown_df=st.session_state.get('correlation_breakdown_df'),
                            predictions_data=ml_predictions  # 머신러닝 모드에서 저장한 예측 결과
                        )
                        
                        if extreme_enhanced_model_data:
                            st.success("Enhanced data with extreme events created successfully!")
                            
                            # 요약 생성 - generate_executive_summary 함수 호출
                            summary = generate_executive_summary(
                                extreme_events_df=st.session_state['extreme_events_df'],
                                black_swan_df=st.session_state['black_swan_df'],
                                technical_patterns_df=st.session_state.get('technical_patterns_df'),
                                correlation_breakdown_df=st.session_state.get('correlation_breakdown_df'),
                                daily_avg_prices=daily_avg_prices,
                                daily_indicators=daily_indicators,
                                extreme_enhanced_model_data=extreme_enhanced_model_data,  # 중요: 여기에 생성된 extreme_enhanced_model_data를 전달
                                categories=daily_avg_prices.columns.tolist()
                            )
                            
                            # 요약 대시보드 표시
                            display_executive_dashboard(
                                summary=summary,
                                daily_avg_prices=daily_avg_prices,
                                daily_indicators=daily_indicators,
                                black_swan_df=st.session_state['black_swan_df'],
                                extreme_events_df=st.session_state['extreme_events_df']
                            )
                        else:
                            st.error("Failed to create enhanced data with extreme events.")
                except Exception as e:
                    st.error(f"Error generating insights and report: {str(e)}")
                    st.exception(e)
            else:
                st.info("Please detect extreme events first to generate insights and report.")
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import norm
def generate_model_results_dashboard():
    """모델 학습 결과를 종합적으로 보여주는 대시보드"""
    st.header("🧮 ML Model Result Dashboard")
    
    # 세션 상태에서 성능 데이터 확인
    if ('model_results' not in st.session_state or 
        not st.session_state['model_results'] or 
        'enhanced_model_data' not in st.session_state):
        st.error("⚠️ No model performance data available. Please run the analysis first.")
        return
    
    # 모델 결과에서 성능 데이터 추출
    model_results = st.session_state['model_results']
    categories = list(model_results.keys())
    
    # 성능 데이터 프레임 생성
    performance_data = []
    for category, result in model_results.items():
        if result:  # 결과가 있는 경우만 처리
            performance_data.append({
                "Category": category,
                "Model Type": result.get('model_name', 'Unknown'),
                "Accuracy": result.get('accuracy', 0),
                "Optimal Threshold Accuracy": result.get('accuracy_optimal', 0),
                "ROC-AUC": result.get('roc_auc', 0),
                "Prediction Confidence": result.get('roc_auc', 0) * 0.5 + result.get('accuracy_optimal', 0) * 0.5  # 예측 신뢰도 점수
            })
    
    # 성능 데이터프레임 생성
    performance_df = pd.DataFrame(performance_data)
    
    # 성능 데이터프레임 저장
    st.session_state['full_performance'] = performance_df
    
    # 전체 성능 테이블
    st.subheader("📋 Overall Model Performance Summary")
    st.dataframe(performance_df.style.format({
        "Accuracy": "{:.2%}",
        "Optimal Threshold Accuracy": "{:.2%}",
        "ROC-AUC": "{:.2%}",
        "Prediction Confidence": "{:.2%}"
    }))
    
    # 성능 비교 바 차트
    try:
        st.subheader("📊 Model Performance Comparison Across Categories")
        fig, ax = plt.subplots(figsize=(10, 6))
        chart_df = performance_df.melt(
            id_vars=["Category", "Model Type"],
            value_vars=["Accuracy", "Optimal Threshold Accuracy", "ROC-AUC"],
            var_name="Metric",
            value_name="Score"
        )
        import seaborn as sns
        sns.barplot(x="Category", y="Score", hue="Metric", data=chart_df, ax=ax)
        plt.title("Category-wise Model Performance", fontsize=16)
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Error generating performance chart: {e}")
    
    st.markdown("---")
    
    # 카테고리별 세부 결과
    st.subheader("🔎 Category-wise Detailed Results")
    selected_category = st.selectbox("Select a category to view detailed results", categories)
    
    if selected_category and selected_category in model_results:
        # 시각화 데이터 가져오기
        if 'visualizations' in st.session_state and selected_category in model_results:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🎯 Feature Importance - {selected_category}")
                if f"{selected_category}_feature_importance" in st.session_state['visualizations']:
                    st.pyplot(st.session_state['visualizations'][f"{selected_category}_feature_importance"])
                else:
                    st.info("No feature importance visualization available.")
                
                st.markdown(f"### 📈 ROC Curve - {selected_category}")
                if f"{selected_category}roc_curve" in st.session_state['visualizations']:
                    st.pyplot(st.session_state['visualizations'][f"{selected_category}roc_curve"])
                else:
                    st.info("No ROC curve visualization available.")
            
            with col2:
                st.markdown(f"### 🧩 Confusion Matrix - {selected_category}")
                if f"{selected_category}_Confusion Matrix" in st.session_state['visualizations']:
                    st.pyplot(st.session_state['visualizations'][f"{selected_category}_Confusion Matrix"])
                else:
                    st.info("No confusion matrix visualization available.")
                
                st.markdown(f"### 🧮 Prediction Probability Distribution - {selected_category}")
                if f"{selected_category}probability_distribution" in st.session_state['visualizations']:
                    st.pyplot(st.session_state['visualizations'][f"{selected_category}probability_distribution"])
                else:
                    st.info("No probability distribution visualization available.")
        else:
            # 세션에 시각화 데이터가 없는 경우 파일 시도
            output_dir = "ML_results"  # 파일이 저장된 디렉토리
            category_dir = os.path.join(output_dir, 'categories', selected_category)
            
            if os.path.exists(category_dir):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### 🎯 Feature Importance - {selected_category}")
                    feature_img_path = os.path.join(category_dir, 'feature_importance.png')
                    if os.path.exists(feature_img_path):
                        st.image(feature_img_path)
                    else:
                        st.info("No feature importance image available.")
                    
                    st.markdown(f"### 📈 ROC Curve - {selected_category}")
                    roc_img_path = os.path.join(category_dir, 'roc_curve.png')
                    if os.path.exists(roc_img_path):
                        st.image(roc_img_path)
                    else:
                        st.info("No ROC curve image available.")
                
                with col2:
                    st.markdown(f"### 🧩 Confusion Matrix - {selected_category}")
                    cm_img_path = os.path.join(category_dir, 'confusion_matrix.png')
                    if os.path.exists(cm_img_path):
                        st.image(cm_img_path)
                    else:
                        st.info("No confusion matrix image available.")
                    
                    st.markdown(f"### 🧮 Prediction Probability Distribution - {selected_category}")
                    prob_img_path = os.path.join(category_dir, 'probability_distribution.png')
                    if os.path.exists(prob_img_path):
                        st.image(prob_img_path)
                    else:
                        st.info("No probability distribution image available.")
                
                # Temporal trend 추가
                st.markdown(f"### ⏳ Temporal Prediction Trend - {selected_category}")
                temporal_img_path = os.path.join(category_dir, 'temporal_prediction.png')
                if os.path.exists(temporal_img_path):
                    st.image(temporal_img_path)
                else:
                    st.info("No temporal trend image available.")
            else:
                st.warning(f"No results directory found for {selected_category}")
    else:
        st.info("Please select a category to view detailed results")

def calculate_correlations(data, category, feature_cols, num_indicators):
    """
    주어진 데이터에서 특정 카테고리와 다른 변수 간의 상관관계를 계산합니다.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        상관관계를 계산할 데이터프레임
    category : str
        상관관계를 계산할 대상 카테고리(컬럼)
    feature_cols : list
        상관관계를 계산할 특성 컬럼 목록
    num_indicators : int
        반환할 상위 상관관계 수
    
    Returns:
    --------
    pandas.Series
        상위 N개 상관관계 (절대값 기준으로 정렬됨)
    """
    try:
        # 카테고리 컬럼이 데이터에 있는지 확인
        if category not in data.columns:
            return pd.Series()
            
        # 특성 컬럼이 데이터에 있는지 확인
        valid_features = [col for col in feature_cols if col in data.columns]
        if not valid_features:
            return pd.Series()
        
        # 상관관계 계산
        correlations = data[valid_features].corrwith(data[category]).abs().sort_values(ascending=False)
        
        # 상위 N개 반환
        return correlations.head(num_indicators)
    except Exception as e:
        print(f"상관관계 계산 중 오류: {str(e)}")
        return pd.Series()
def generate_market_insights_dashboard(enhanced_model_data, daily_avg_prices, daily_indicators):
    """
    시장 인사이트를 보여주는 대시보드를 생성합니다.
    각 카테고리마다 별도의 탭을 제공하고 예측 기능을 개선합니다.
    
    Args:
        enhanced_model_data: 향상된 모델 데이터 (카테고리별 데이터프레임)
        daily_avg_prices: 일별 평균 가격 데이터
        daily_indicators: 일별 지표 데이터
    """
    st.header("🧠 Market Insights Dashboard")

    if 'enhanced_model_data' not in st.session_state:
        st.session_state['enhanced_model_data'] = enhanced_model_data
    if not enhanced_model_data:
        st.error("No model data available. Please run the analysis first.")
        return
    
    # 세션 상태 초기화: 현재 파라미터 저장
    if 'market_dashboard' not in st.session_state:
        st.session_state['market_dashboard'] = {}
    
    # 카테고리별 데이터 캐싱 (계산 결과 저장)
    if 'category_cache' not in st.session_state:
        st.session_state['category_cache'] = {}
    
    # 카테고리별 시장 인사이트 저장을 위한 세션 상태 확인
    if 'market_insights' not in st.session_state:
        st.session_state['market_insights'] = {}

    # 3. 카테고리 탭 관련 세션 초기화
    categories = list(enhanced_model_data.keys())
    if not categories:
        st.warning("No categories available for analysis.")
        return

    if 'selected_category_tab' not in st.session_state:
        st.session_state['selected_category_tab'] = 0

    selected_tab = st.session_state['selected_category_tab']
    category_tabs = st.tabs([f"📊 {category}" for category in categories])

    # 4. 각 카테고리 탭 내부
    for i, category in enumerate(categories):
        with category_tabs[i]:
            # 🔥 여기서부터 category 사용할 수 있다

            if category not in st.session_state['market_dashboard']:
                st.session_state['market_dashboard'][category] = {
                    'date_range': None,
                    'ma_days': 20,
                    'num_indicators': 10
                }

            if category not in st.session_state['market_insights']:
                st.session_state['market_insights'][category] = {}

            if 'selected_dashboard_tab' not in st.session_state:
                st.session_state['selected_dashboard_tab'] = {}

            if category not in st.session_state['selected_dashboard_tab']:
                st.session_state['selected_dashboard_tab'][category] = 0

            # 5. 데이터 체크
            category_data = enhanced_model_data[category]
            if category_data is None or len(category_data) == 0:
                st.warning(f"No data available for {category}")
                continue

            if 'date' not in category_data.columns and isinstance(category_data.index, pd.DatetimeIndex):
                category_data = category_data.reset_index()
                category_data.rename(columns={'index': 'date'}, inplace=True)

            # 각 카테고리별 분석 탭 생성
            dashboard_tabs = st.tabs([
                "📈 Price Analysis", 
                "🔄 Correlation Analysis", 
                "🔮 Prediction Analysis",
                "💡 Insights Summary"
            ])
            selected_tab_index = st.session_state.get('selected_dashboard_tab', 0)
            # 3. 가격 분석 탭
            with dashboard_tabs[0]:
                st.subheader(f"📈 Price Trend Analysis: {category}")
                
                try:
                    # 가격 데이터 추출
                    if category in daily_avg_prices.columns:
                        price_series = daily_avg_prices[category].dropna()
                        
                        # 세션에 가격 데이터 저장
                        st.session_state['market_insights'][category]['price_data'] = price_series.to_dict()
                        
                        # 날짜 범위 선택기
                        if not price_series.empty and isinstance(price_series.index, pd.DatetimeIndex):
                            min_date = price_series.index.min().date()
                            max_date = price_series.index.max().date()
                            
                            # 날짜 범위 기본값 설정
                            col1, col2 = st.columns(2)
                            with col1:
                                start_date = st.date_input(
                                    "📅 Start Date",
                                    value=max(min_date, max_date - pd.Timedelta(days=180)),
                                    min_value=min_date,
                                    max_value=max_date,
                                    key=f"start_date_{category}"
                                )
                            with col2:
                                end_date = st.date_input(
                                    "📅 End Date",
                                    value=max_date,
                                    min_value=min_date,
                                    max_value=max_date,
                                    key=f"end_date_{category}"
                                )
                            
                            # 세션에 날짜 범위 저장
                            date_range = (start_date, end_date)
                            st.session_state['market_dashboard'][category]['date_range'] = date_range
                            
                            # 날짜 범위 필터링
                            mask = (price_series.index.date >= start_date) & (price_series.index.date <= end_date)
                            filtered_prices = price_series[mask]
                            
                            # 캐시 키 생성 (카테고리, 날짜 범위, MA 윈도우)
                            cache_key = f"{category}_price_{start_date}_{end_date}"
                            
                            # 이동 평균 윈도우
                            ma_days = st.slider(
                                "📊 Moving Average Window (days)", 
                                5, 50, 
                                st.session_state['market_dashboard'][category]['ma_days'],
                                key=f"ma_days_slider_{category}"
                            )
                            st.session_state['market_dashboard'][category]['ma_days'] = ma_days
                            
                            # 필터링된 데이터가 있는 경우
                            if not filtered_prices.empty:
                                # 캐시에 없으면 계산하고 저장
                                if cache_key not in st.session_state['category_cache']:
                                    price_stats = compute_price_statistics(filtered_prices)
                                    st.session_state['category_cache'][cache_key] = price_stats
                                else:
                                    price_stats = st.session_state['category_cache'][cache_key]
                                
                                # 가격 차트와 통계 표시를 위한 열 레이아웃
                                price_chart_col, price_stats_col = st.columns([3, 1])
                                
                                with price_chart_col:
                                    # 가격 트렌드 차트 생성
                                    price_chart_fig = create_price_trend_chart(
                                        filtered_prices, 
                                        category, 
                                        ma_days
                                    )
                                    st.plotly_chart(price_chart_fig, use_container_width=True)
                                    
                                with price_stats_col:
                                    # 가격 통계 표시
                                    current_price = filtered_prices.iloc[-1]
                                    price_change = (filtered_prices.iloc[-1] - filtered_prices.iloc[0]) / filtered_prices.iloc[0]
                                    volatility = filtered_prices.pct_change().std() * (252 ** 0.5)
                                    sharpe = (filtered_prices.pct_change().mean() / filtered_prices.pct_change().std()) * (252 ** 0.5) if filtered_prices.pct_change().std() != 0 else 0
                                    
                                    st.metric("Current Price", f"{current_price:.2f}")
                                    st.metric("Period Change", f"{price_change:.2%}", 
                                            delta=f"{price_change:.2%}", 
                                            delta_color="normal")
                                    st.metric("Volatility (Annual)", f"{volatility:.2%}")
                                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                                    
                                    # 통계 정보 세션에 저장
                                    st.session_state['market_insights'][category]['stats'] = {
                                        'current_price': float(current_price),
                                        'price_change': float(price_change),
                                        'volatility': float(volatility),
                                        'sharpe': float(sharpe)
                                    }
                                
                                # 추가 분석: 가격 분포 및 수익률 분포
                                st.subheader("📊 Price & Return Distribution")
                                dist_col1, dist_col2 = st.columns(2)
                                
                                with dist_col1:
                                    # 가격 분포 히스토그램
                                    price_hist_fig = create_price_histogram(filtered_prices, category)
                                    st.plotly_chart(price_hist_fig, use_container_width=True)
                                    
                                with dist_col2:
                                    # 수익률 분포 히스토그램
                                    returns = filtered_prices.pct_change().dropna()
                                    returns_hist_fig = create_returns_histogram(returns, category)
                                    st.plotly_chart(returns_hist_fig, use_container_width=True)
                            else:
                                st.warning("No price data available for the selected date range")
                        else:
                            st.warning("Price data is empty or does not have a proper datetime index")
                    else:
                        st.warning(f"No price data available for {category}")
                except Exception as e:
                    st.error(f"Error while analyzing price trends: {str(e)}")
            
            # 4. 상관관계 분석 탭
            with dashboard_tabs[1]:
                if 'selected_dashboard_tab' not in st.session_state:
                    st.session_state['selected_dashboard_tab'] = {}
                    st.session_state['selected_dashboard_tab'][category] = 1 
                st.subheader(f"🔄 Indicator Correlation Analysis: {category}")
                
                try:
                    # 선택된 범주의 데이터에서 관련 지표 찾기
                    feature_cols = [col for col in category_data.columns if (
                        col != category and 
                        col != 'date' and 
                        col != 'predictions' and
                        not col.startswith('lag_')
                    )]
                    
                    if not feature_cols:
                        st.warning("No indicator features found for correlation analysis")
                        return
            
                    # 상위 상관관계 지표 선택
                    # 개선된 코드 - 슬라이더 상태 관리
                    if 'market_dashboard' not in st.session_state:
                        st.session_state['market_dashboard'] = {}
                    
                    if category not in st.session_state['market_dashboard']:
                        st.session_state['market_dashboard'][category] = {
                            'num_indicators': 10  # 기본값
                        }
                    
                    slider_key = f"num_indicators_slider_{category}"
                    
                    with st.form(key=f"indicator_form_{category}"):
                        num_indicators = st.slider(
                            "📏 Number of Indicators to Show", 
                            5, min(20, len(feature_cols)), 
                            st.session_state['market_dashboard'][category]['num_indicators'],
                            key=slider_key
                        )
                        
                        submit_button = st.form_submit_button("Apply")
                        
                        if submit_button:
                            st.session_state['market_dashboard'][category]['num_indicators'] = num_indicators
                            st.session_state['selected_dashboard_tab'] = 1  # Correlation Analysis 탭
                            st.experimental_rerun()
                        # 값이 변경되면 페이지 새로고침
                     # 라이더 결과(num_indicators)를 꺼내오는 거야.
                    num_indicators = st.session_state['market_dashboard'][category]['num_indicators']
                    # 상관관계 계산 - feature_cols 정의 후에 계산
                    if category in category_data.columns:
                        # 캐시 키 생성
                        sorted_features = sorted(feature_cols)
                        feature_hash = hash(str(sorted_features))
                        corr_cache_key = f"{category}_corr_{num_indicators}_{hash(str(sorted(feature_cols)))}"

                        if 'category_cache' not in st.session_state:
                            st.session_state['category_cache'] = {}

                        # num_indicators까지 키에 반영해 새로 체크
                        need_recompute = (corr_cache_key not in st.session_state['category_cache'])

                        if need_recompute:
                            with st.spinner(f"Calculating correlations for {category}..."):
                                top_correlations = calculate_correlations(category_data, category, feature_cols, num_indicators)
                                if not top_correlations.empty:
                                    st.session_state['category_cache'][corr_cache_key] = top_correlations.copy()
                                    st.success(f"Correlation calculation completed for {category}")
                                else:
                                    st.warning("No correlations found")
                        else:
                            top_correlations = st.session_state['category_cache'][corr_cache_key].copy()
                      
                        # 상관관계 결과가 있는 경우만 시각화
                        if not top_correlations.empty:
                            # 세션에 상관관계 데이터 저장
                            correlation_dict = top_correlations.to_dict()
                            
                            if 'market_insights' not in st.session_state:
                                st.session_state['market_insights'] = {}
                            if category not in st.session_state['market_insights']:
                                st.session_state['market_insights'][category] = {}
                            
                            st.session_state['market_insights'][category]['correlations'] = correlation_dict
                            
                            # 상관관계 시각화
                            corr_chart_col, corr_details_col = st.columns([3, 2])
                            
                            with corr_chart_col:
                                corr_fig = create_correlation_chart(top_correlations, category)
                                st.plotly_chart(corr_fig, use_container_width=True)
                            
                            with corr_details_col:
                                st.markdown("### 📊 Top Correlated Indicators")
                                for i, (feature, corr_value) in enumerate(top_correlations.items()):
                                    corr_color = "green" if corr_value > 0.7 else ("orange" if corr_value > 0.5 else "gray")
                                    st.markdown(f"{i+1}. **{feature}**: <span style='color:{corr_color};font-weight:bold'>{corr_value:.3f}</span>", unsafe_allow_html=True)
                            
                            # 상관관계 히트맵
                            st.markdown("### 🔥 Correlation Heatmap")
                            
                            # 상위 지표와 가격 데이터 결합
                            heatmap_cols = top_correlations.index.tolist() + [category]
                            if all(col in category_data.columns for col in heatmap_cols):
                                # 히트맵 캐시 키
                                heatmap_cache_key = f"{category}_heatmap_{num_indicators}_{hash(str(feature_cols))}"
                                
                                # 캐시에 없으면 계산
                                if heatmap_cache_key not in st.session_state['category_cache']:
                                    corr_matrix = category_data[heatmap_cols].corr()
                                    st.session_state['category_cache'][heatmap_cache_key] = corr_matrix
                                else:
                                    corr_matrix = st.session_state['category_cache'][heatmap_cache_key]
                                
                                # 세션에 상관관계 행렬 저장
                                st.session_state['market_insights'][category]['correlation_matrix'] = corr_matrix.to_dict()
                                
                                # Plotly 히트맵 생성
                                heatmap_fig = create_correlation_heatmap(corr_matrix, category)
                                st.plotly_chart(heatmap_fig, use_container_width=True)
                                
                                # 인사이트 생성
                                st.subheader("💡 Correlation Insights")
                                pos_correlations = top_correlations[top_correlations > 0.5]
                                neg_correlations = top_correlations[top_correlations < -0.5]
                                
                                if not pos_correlations.empty:
                                    st.info(f"**Positive Relationships**: {category} shows strong positive correlation with {', '.join(pos_correlations.index[:3])}. These indicators tend to move in the same direction as the asset price.")
                                    
                                if not neg_correlations.empty:
                                    st.warning(f"**Negative Relationships**: {category} shows strong negative correlation with {', '.join(neg_correlations.index[:3])}. These indicators tend to move in the opposite direction to the asset price.")
                            else:
                                st.warning("Some selected columns are not available in the data")
                        else:
                            st.warning("No significant correlations found")
                    else:
                        st.warning(f"{category} column not found in the data")
                except Exception as e:
                    st.error(f"Error during correlation analysis: {str(e)}")
            
            # 5. 예측 분석 탭
            with dashboard_tabs[2]:
                st.subheader(f"🔮 Prediction Pattern Analysis: {category}")
                
                try:
                    # 예측 결과가 있는지 확인
                    prediction_data = category_data.copy()
                    
                    # prediction_data 검사
                    has_predictions = False
                    if 'predictions' in prediction_data.columns:
                        if prediction_data['predictions'].notna().sum() > 0:
                            has_predictions = True
                            st.success(f"Found {prediction_data['predictions'].notna().sum()} valid predictions in the data.")
                    
                    # 예측값 없으면 생성 (디버깅 정보 출력)
                    if not has_predictions:
                        st.warning("No predictions found or all are NaN. Checking data structure...")
                        
                        # 디버깅 정보
                        st.expander("Debugging Information").write({
                            "Columns in category_data": category_data.columns.tolist(),
                            "Has predictions column": 'predictions' in category_data.columns,
                            "Valid predictions count": category_data['predictions'].notna().sum() if 'predictions' in category_data.columns else 0,
                            "Has price column": category in category_data.columns,
                            "Data shape": category_data.shape,
                            "First few rows": category_data.head().to_dict() if not category_data.empty else "Empty DataFrame"
                        })
                        
                        # 예측 데이터 생성 (가상 예측)
                        st.info("Generating synthetic predictions for demonstration...")
                        
                        if category in prediction_data.columns:
                            # 실제 방향을 기반으로 예측 생성 (약간의 오차 추가)
                            actual_dir = (prediction_data[category].diff() > 0).astype(int)
                            
                            # 80% 정확도를 가진 가상 예측 생성
                            np.random.seed(42)  # 재현성을 위한 시드 설정
                            random_flip = np.random.random(len(actual_dir)) > 0.8  # 20% 확률로 뒤집기
                            prediction_data['predictions'] = np.where(random_flip, 1 - actual_dir, actual_dir)
                            
                            # NaN 처리
                            prediction_data['predictions'] = prediction_data['predictions'].fillna(0)
                            has_predictions = True
                    
                    if has_predictions:
                        # 예측 데이터 처리
                        if 'date' in prediction_data.columns or isinstance(prediction_data.index, pd.DatetimeIndex):
                            # 날짜 형식 확인 및 처리
                            if 'date' in prediction_data.columns and not pd.api.types.is_datetime64_any_dtype(prediction_data['date']):
                                prediction_data['date'] = pd.to_datetime(prediction_data['date'])
                                prediction_data.set_index('date', inplace=True)
                            elif not isinstance(prediction_data.index, pd.DatetimeIndex) and 'date' not in prediction_data.columns:
                                st.warning("No date information available for prediction analysis")
                                return
                            
                            # 최근 예측 결과 (마지막 30개 또는 전체 중 작은 값)
                            recent_count = min(30, len(prediction_data))
                            recent_predictions = prediction_data.iloc[-recent_count:]
                            
                            pred_chart_col, pred_metrics_col = st.columns([3, 1])
                            
                            with pred_chart_col:
                                # 예측 트렌드 차트 (Plotly로 개선)
                                if category in recent_predictions.columns and 'predictions' in recent_predictions.columns:
                                    pred_fig = create_prediction_chart(recent_predictions, category)
                                    st.plotly_chart(pred_fig, use_container_width=True)
                            
                            with pred_metrics_col:
                                # 예측 정확도 및 지표
                                if 'predictions' in recent_predictions.columns and category in recent_predictions.columns:
                                    # 실제 방향 계산 (당일 종가와 전일 종가 비교)
                                    actual_direction = (recent_predictions[category].diff() > 0).astype(int)
                                    # NaN 값 처리
                                    actual_direction = actual_direction.fillna(0)
                                    
                                    # 정확도 계산 (NaN 값 제외)
                                    valid_indices = (~actual_direction.isna()) & (~recent_predictions['predictions'].isna())
                                    if valid_indices.sum() > 0:
                                        correct_predictions = (actual_direction[valid_indices] == recent_predictions['predictions'][valid_indices]).mean()
                                    else:
                                        correct_predictions = 0
                                    
                                    # 세션에 최근 예측 정확도 저장
                                    st.session_state['market_insights'][category]['recent_accuracy'] = float(correct_predictions)
                                    
                                    st.metric("Recent Accuracy", f"{correct_predictions:.2%}")
                                    
                                    # 상승/하락 예측 비율
                                    up_ratio = recent_predictions['predictions'].mean()
                                    st.metric("Up Prediction Ratio", f"{up_ratio:.2%}")
                                    
                                    # 최근 예측 방향
                                    latest_pred = "Up" if recent_predictions['predictions'].iloc[-1] == 1 else "Down"
                                    pred_color = "green" if latest_pred == "Up" else "red"
                                    st.markdown(f"**Latest Prediction**: <span style='color:{pred_color};font-weight:bold'>{latest_pred}</span>", unsafe_allow_html=True)
                            
                            # 예측 성과 지표
                            st.subheader("📊 Prediction Performance Analysis")
                            
                            if 'predictions' in prediction_data.columns and category in prediction_data.columns:
                                perf_col1, perf_col2 = st.columns(2)
                                
                                with perf_col1:
                                    # 실제 방향과 예측 방향 계산
                                    # 수정: 실제 방향은 전일 대비 변화로 계산
                                    actual_direction = (prediction_data[category].diff() > 0).astype(int)
                                    actual_direction = actual_direction.fillna(0)  # 첫 행의 NaN 처리
                                    
                                    # 혼동 행렬 계산 (전체 데이터셋 기준)
                                    actual_up = actual_direction.sum()
                                    actual_down = len(actual_direction) - actual_up
                                    predicted_up = prediction_data['predictions'].sum()
                                    predicted_down = len(prediction_data['predictions']) - predicted_up
                                    
                                    # 예측 성능 지표
                                    valid_indices = (~actual_direction.isna()) & (~prediction_data['predictions'].isna())
                                    if valid_indices.sum() > 0:
                                        correct_predictions = (actual_direction[valid_indices] == prediction_data['predictions'][valid_indices]).mean()
                                    else:
                                        correct_predictions = 0
                                    
                                    # 정밀도 계산 (예측이 상승일 때 실제로 상승인 비율)
                                    up_indices = prediction_data['predictions'] == 1
                                    if up_indices.sum() > 0:
                                        up_precision = (actual_direction[up_indices] == 1).mean()
                                    else:
                                        up_precision = 0
                                    
                                    # 정밀도 계산 (예측이 하락일 때 실제로 하락인 비율)
                                    down_indices = prediction_data['predictions'] == 0
                                    if down_indices.sum() > 0:
                                        down_precision = (actual_direction[down_indices] == 0).mean()
                                    else:
                                        down_precision = 0
                                    
                                    # 세션에 예측 성능 지표 저장
                                    st.session_state['market_insights'][category]['prediction_metrics'] = {
                                        'accuracy': float(correct_predictions),
                                        'up_precision': float(up_precision),
                                        'down_precision': float(down_precision),
                                        'coverage': float(len(prediction_data[~prediction_data['predictions'].isna()]) / len(prediction_data))
                                    }
                                    
                                    # 성능 지표 표시
                                    metrics_cols = st.columns(2)
                                    with metrics_cols[0]:
                                        st.metric("Overall Accuracy", f"{correct_predictions:.2%}")
                                        st.metric("Up Precision", f"{up_precision:.2%}")
                                    with metrics_cols[1]:
                                        st.metric("Down Precision", f"{down_precision:.2%}")
                                        st.metric("Prediction Coverage", f"{len(prediction_data[~prediction_data['predictions'].isna()]) / len(prediction_data):.2%}")
                                    
                                    # 실제 vs 예측 차트 (Plotly로 개선)
                                    labels = ['Down', 'Up']
                                    actual_counts = [actual_down, actual_up]
                                    predicted_counts = [predicted_down, predicted_up]
                                    
                                    # NaN 값 처리
                                    actual_counts = np.nan_to_num(actual_counts)
                                    predicted_counts = np.nan_to_num(predicted_counts)
                                    
                                    direction_fig = create_direction_counts_chart(labels, actual_counts, predicted_counts)
                                    st.plotly_chart(direction_fig, use_container_width=True)
                                
                                with perf_col2:
                                    # 혼동 행렬 시각화
                                    from sklearn.metrics import confusion_matrix
                                    
                                    # 유효한 인덱스만 사용
                                    valid_mask = ~actual_direction.isna() & ~prediction_data['predictions'].isna()
                                    if valid_mask.sum() > 1:  # 최소 2개 이상의 데이터 필요
                                        # 혼동 행렬 계산
                                        cm = confusion_matrix(
                                            actual_direction[valid_mask], 
                                            prediction_data['predictions'][valid_mask]
                                        )
                                        
                                        # 혼동 행렬 시각화
                                        cm_fig = create_confusion_matrix_chart(cm, category)
                                        st.plotly_chart(cm_fig, use_container_width=True)
                                    else:
                                        st.warning("Not enough valid data for confusion matrix")
                                    
                                    # 시간 경과에 따른 정확도 추이
                                    if len(prediction_data) > 10:
                                        # 윈도우 크기 계산 (최소 10개 데이터 포인트)
                                        window_size = max(10, len(prediction_data) // 10)
                                        
                                        # 롤링 정확도 계산 (유효한 값만 사용)
                                        valid_mask = ~actual_direction.isna() & ~prediction_data['predictions'].isna()
                                        if valid_mask.sum() > window_size:
                                            # 정확/부정확 시리즈 생성 (1: 정확, 0: 부정확)
                                            accuracy_series = pd.Series(
                                                (actual_direction == prediction_data['predictions']).astype(float),
                                                index=prediction_data.index
                                            )
                                            
                                            # 롤링 정확도 계산
                                            rolling_accuracy = accuracy_series.rolling(window=window_size).mean()
                                            
                                            # 정확도 추이 차트
                                            accuracy_trend_fig = create_accuracy_trend_chart(rolling_accuracy, window_size, category)
                                            st.plotly_chart(accuracy_trend_fig, use_container_width=True)
                                        else:
                                            st.warning(f"Not enough valid data points for rolling accuracy (need at least {window_size} valid points)")
                            else:
                                st.warning("Prediction data not available for performance metrics")
                        else:
                            st.warning("No date information available for the prediction analysis")
                    else:
                        st.info("No valid predictions found. Run the analysis to generate predictions.")
                except Exception as e:
                    st.error(f"Error during prediction pattern analysis: {str(e)}")
                    st.exception(e)
            
            # 6. 인사이트 요약 탭
            with dashboard_tabs[3]:
                st.subheader(f"💡 Market Insights Summary: {category}")
                
                try:
                    if 'predictions' in category_data.columns and category in category_data.columns:
                        # 최근 20거래일 예측 데이터
                        recent_data = category_data.iloc[-20:].copy() if len(category_data) >= 20 else category_data.copy()
                        
                        # 실제 방향과 예측 방향
                        if category in recent_data.columns:
                            recent_data['actual_direction'] = np.where(recent_data[category].diff() > 0, 'Up', 'Down')
                        
                        if 'predictions' in recent_data.columns:
                            recent_data['predicted_direction'] = np.where(recent_data['predictions'] == 1, 'Up', 'Down')
                        
                        # 세션에 요약 데이터 저장
                        if 'market_insights' not in st.session_state:
                            st.session_state['market_insights'] = {}
                        if category not in st.session_state['market_insights']:
                            st.session_state['market_insights'][category] = {}
                        if 'recent_summary' not in st.session_state['market_insights'][category]:
                            st.session_state['market_insights'][category]['recent_summary'] = {}
                        
                        # 최근 예측 요약
                        if 'predicted_direction' in recent_data.columns:
                            # 데이터 분석 결과 요약
                            summary_col1, summary_col2 = st.columns([2, 1])
                            
                            with summary_col1:
                                # 전체 인사이트 요약 카드
                                st.markdown("""
                                <style>
                                .insight-card {
                                    border: 1px solid #ddd;
                                    border-radius: 5px;
                                    padding: 20px;
                                    background-color: #f9f9f9;
                                    margin-bottom: 20px;
                                    color: black;
                                }
                                .insight-title {
                                    font-size: 18px;
                                    font-weight: bold;
                                    margin-bottom: 10px;
                                    color: black;
                                }
                                </style>
                                """, unsafe_allow_html=True)
                                
                                latest_prediction = recent_data['predicted_direction'].iloc[-1] if not recent_data.empty else 'Unknown'
                                prediction_trend = recent_data['predicted_direction'].value_counts().idxmax() if not recent_data.empty else 'Unknown'
                                
                                # 세션에 최근 예측 요약 저장
                                st.session_state['market_insights'][category]['recent_summary']['latest_prediction'] = latest_prediction
                                st.session_state['market_insights'][category]['recent_summary']['prediction_trend'] = prediction_trend
                                
                                trend_color = "green" if prediction_trend == "Up" else "red"
                                trend_icon = "📈" if prediction_trend == "Up" else "📉"
                                
                                st.markdown(f"""
                                <div class="insight-card">
                                    <div class="insight-title">{trend_icon} Market Trend Analysis</div>
                                    <p>The model indicates a <span style='color:{trend_color};font-weight:bold'>{prediction_trend.lower()}</span> trend for <b>{category}</b>. Recent predictions suggest {prediction_trend.lower()}ward price movements are more likely in the near term.</p>
                                    <p>Latest prediction: <span style='color:{trend_color};font-weight:bold'>{latest_prediction}</span></p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # 성능 인사이트
                                if 'prediction_metrics' in st.session_state['market_insights'][category]:
                                    metrics = st.session_state['market_insights'][category]['prediction_metrics']
                                    accuracy = metrics.get('accuracy', 0)
                                    reliability = "High" if accuracy > 0.7 else ("Moderate" if accuracy > 0.5 else "Low")
                                    rel_color = "green" if reliability == "High" else ("orange" if reliability == "Moderate" else "red")
                                    
                                    st.markdown(f"""
                                    <div class="insight-card">
                                        <div class="insight-title">🎯 Model Reliability Assessment</div>
                                        <p>Model reliability: <span style='color:{rel_color};font-weight:bold'>{reliability}</span> ({accuracy:.2%} accuracy)</p>
                                        <p>The model shows {metrics.get('up_precision', 0):.2%} precision for upward movements and {metrics.get('down_precision', 0):.2%} precision for downward movements.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # 상관관계 인사이트
                                if 'correlations' in st.session_state['market_insights'][category]:
                                    correlations = pd.Series(st.session_state['market_insights'][category]['correlations'])
                                    top_pos = correlations[correlations > 0].nlargest(3)
                                    top_neg = correlations[correlations < 0].nsmallest(3)
                                    
                                    corr_html = "<div class='insight-card'><div class='insight-title'>🔄 Key Correlations</div><p>"
                                    
                                    if not top_pos.empty:
                                        corr_html += f"<b>Top positive indicators</b>: "
                                        for i, (ind, val) in enumerate(top_pos.items()):
                                            corr_html += f"{ind} ({val:.2f})" + (", " if i < len(top_pos) - 1 else "")
                                    
                                    if not top_neg.empty:
                                        corr_html += f"</p><p><b>Top negative indicators</b>: "
                                        for i, (ind, val) in enumerate(top_neg.items()):
                                            corr_html += f"{ind} ({val:.2f})" + (", " if i < len(top_neg) - 1 else "")
                                    
                                    corr_html += "</p></div>"
                                    st.markdown(corr_html, unsafe_allow_html=True)
                            
                            with summary_col2:
                                # 최근 5일 예측 동향
                                st.markdown("### 📅 Recent Predictions")
                                
                                last_5_days = recent_data.iloc[-5:]['predicted_direction'].tolist() if len(recent_data) >= 5 else recent_data['predicted_direction'].tolist()
                                
                                # 세션에 최근 5일 예측 동향 저장
                                st.session_state['market_insights'][category]['recent_summary']['last_5_days'] = last_5_days
                                
                                # 최근 예측 시각화 - 향상된 디자인
                                for i, pred in enumerate(reversed(last_5_days)):
                                    days_ago = len(last_5_days) - 1 - i
                                    day_text = "Today" if days_ago == 0 else f"{days_ago} day{'s' if days_ago > 1 else ''} ago"
                                    icon = "📈" if pred == "Up" else "📉"
                                    color = "green" if pred == "Up" else "red"
                                    bg_color = "#e6ffe6" if pred == "Up" else "#ffe6e6"
                                    
                                    st.markdown(f"""
                                    <div style="padding: 10px; margin-bottom: 5px; border-radius: 5px; background-color: {bg_color}; display: flex; justify-content: space-between; color: black;">
                                        <span style="font-weight: bold;">{day_text}</span>
                                        <span>{icon} <span style="color: {color}; font-weight: bold;">{pred}</span></span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # 추가 통계 정보
                                if 'stats' in st.session_state['market_insights'][category]:
                                    stats = st.session_state['market_insights'][category]['stats']
                                    
                                    st.markdown("### 📊 Price Statistics")
                                    st.markdown(f"""
                                    <div style="padding: 10px; margin-bottom: 5px; border-radius: 5px; background-color: #f0f0f0; color: black;">
                                        <span style="font-weight: bold; color: black;">Current Price: </span> {stats.get('current_price', 0):.2f}
                                    </div>
                                    <div style="padding: 10px; margin-bottom: 5px; border-radius: 5px; background-color: #f0f0f0; color: black;">
                                        <span style="font-weight: bold; color: black;">Period Change:</span> {stats.get('price_change', 0):.2%}
                                    </div>
                                    <div style="padding: 10px; margin-bottom: 5px; border-radius: 5px; background-color: #f0f0f0; color: black;">
                                        <span style="font-weight: bold; color: black;">Volatility:</span> {stats.get('volatility', 0):.2%}
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # 예측 캘린더 시각화
                            st.subheader("📆 Prediction Calendar")
                            
                            # 최근 30일 예측을 캘린더 형식으로 표시
                            if len(recent_data) > 0 and 'predicted_direction' in recent_data.columns:
                                calendar_data = recent_data.iloc[-30:] if len(recent_data) >= 30 else recent_data
                                calendar_fig = create_prediction_calendar(calendar_data, category)
                                st.plotly_chart(calendar_fig, use_container_width=True)
                            
                            # 자산 성과 비교 차트
                            if 'price_data' in st.session_state['market_insights'][category] and len(daily_avg_prices.columns) > 1:
                                st.subheader("💹 Asset Performance Comparison")
                                
                                # 비교할 자산 선택 (최대 3개)
                                other_assets = [cat for cat in daily_avg_prices.columns if cat != category]
                                if other_assets:
                                    compare_assets = st.multiselect(
                                        "Select assets to compare",
                                        other_assets,
                                        default=other_assets[:min(2, len(other_assets))],
                                        key=f"compare_assets_{category}"
                                    )
                                    
                                    if compare_assets:
                                        # 비교 차트 생성
                                        comparison_fig = create_asset_comparison_chart(
                                            daily_avg_prices, 
                                            [category] + compare_assets,
                                            start_date if 'market_dashboard' in st.session_state and category in st.session_state['market_dashboard'] and st.session_state['market_dashboard'][category]['date_range'] else None,
                                            end_date if 'market_dashboard' in st.session_state and category in st.session_state['market_dashboard'] and st.session_state['market_dashboard'][category]['date_range'] else None
                                        )
                                        st.plotly_chart(comparison_fig, use_container_width=True)
                        else:
                            st.info("No recent predictions available for trend analysis")
                    else:
                        st.info("No prediction data available. Run the analysis to generate predictions and insights.")
                except Exception as e:
                    st.error(f"Error generating market insights summary: {str(e)}")
                    st.exception(e)
    
    # 세션 상태에 현재 카테고리 업데이트
    if categories:
        st.session_state['current_category'] = categories[0]



# 헬퍼 함수: 가격 통계 계산
def compute_price_statistics(price_series):
    """가격 시계열에 대한 통계 계산"""
    if len(price_series) < 2:
        return {}
    
    # 수익률 계산
    returns = price_series.pct_change().dropna()
    
    # 기본 통계
    stats = {
        'current_price': float(price_series.iloc[-1]),
        'start_price': float(price_series.iloc[0]),
        'price_change': float((price_series.iloc[-1] - price_series.iloc[0]) / price_series.iloc[0]),
        'min_price': float(price_series.min()),
        'max_price': float(price_series.max()),
        'mean_price': float(price_series.mean()),
        'std_price': float(price_series.std()),
        
        # 수익률 통계
        'mean_return': float(returns.mean()),
        'std_return': float(returns.std()),
        'skew_return': float(returns.skew()) if len(returns) > 2 else 0,
        'kurtosis_return': float(returns.kurtosis()) if len(returns) > 3 else 0,
        
        # 변동성 (연간화)
        'volatility': float(returns.std() * (252 ** 0.5)),
        
        # 성과 지표
        'sharpe': float((returns.mean() / returns.std()) * (252 ** 0.5)) if returns.std() != 0 else 0,
    }
    
    return stats

# 헬퍼 함수: Plotly 가격 트렌드 차트 생성
def create_price_trend_chart(price_series, category, ma_days=20):
    """Plotly를 사용한 가격 트렌드 차트 생성"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # 서브플롯 생성 (가격 및 거래량 표시용)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                         vertical_spacing=0.02, row_heights=[0.7, 0.3],
                         subplot_titles=(f"{category} Price Trend", "Returns"))
    
    # 가격 트렌드 라인
    fig.add_trace(
        go.Scatter(
            x=price_series.index, 
            y=price_series.values,
            mode='lines',
            name='Price',
            line=dict(color='royalblue', width=2)
        ),
        row=1, col=1
    )
    
    # 이동 평균선 추가
    if len(price_series) > ma_days:
        ma = price_series.rolling(window=ma_days).mean()
        fig.add_trace(
            go.Scatter(
                x=ma.index, 
                y=ma.values,
                mode='lines',
                name=f'{ma_days}-day MA',
                line=dict(color='firebrick', width=1.5, dash='dot')
            ),
            row=1, col=1
        )
    
    # 수익률 바 차트
    returns = price_series.pct_change().dropna()
    colors = ['green' if x >= 0 else 'red' for x in returns.values]
    
    fig.add_trace(
        go.Bar(
            x=returns.index,
            y=returns.values * 100,  # 퍼센트로 변환
            name='Daily Returns (%)',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # 차트 레이아웃 설정
    fig.update_layout(
        height=600,
        template='plotly_white',
        title=f"{category} Price Analysis",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )
    
    # Y축 레이블 설정
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
    
    return fig
# 헬퍼 함수: 가격 히스토그램 생성
def create_price_histogram(price_series, category):
    """Plotly를 사용한 가격 분포 히스토그램 생성"""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(
            x=price_series.values,
            nbinsx=30,
            name='Price Distribution',
            marker_color='royalblue',
            opacity=0.7
        )
    )
    
    # 현재 가격 표시 라인
    current_price = price_series.iloc[-1]
    
    fig.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="firebrick",
        annotation_text=f"Current: {current_price:.2f}",
        annotation_position="top right"
    )
    
    # 평균 가격 표시 라인
    mean_price = price_series.mean()
    
    fig.add_vline(
        x=mean_price,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Mean: {mean_price:.2f}",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title=f"{category} Price Distribution",
        xaxis_title="Price",
        yaxis_title="Frequency",
        template='plotly_white',
        bargap=0.01
    )
    
    return fig

# 헬퍼 함수: 수익률 히스토그램 생성
def create_returns_histogram(returns, category):
    """Plotly를 사용한 수익률 분포 히스토그램 생성"""
    import plotly.graph_objects as go
    import numpy as np
    
    fig = go.Figure()
    
    # 수익률을 퍼센트로 변환
    returns_pct = returns * 100
    
    fig.add_trace(
        go.Histogram(
            x=returns_pct.values,
            nbinsx=30,
            name='Returns Distribution',
            marker_color='royalblue',
            opacity=0.7
        )
    )
    
    # 정규 분포 오버레이
    x = np.linspace(returns_pct.min(), returns_pct.max(), 100)
    mean = returns_pct.mean()
    std = returns_pct.std()
    
    # 정규 분포 PDF 계산
    from scipy.stats import norm
    pdf = norm.pdf(x, mean, std)
    pdf = pdf * (returns_pct.count() * (returns_pct.max() - returns_pct.min()) / 30)
    
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='firebrick', dash='dash'),
        )
    )
    
    # 0 라인 추가
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="black",
        annotation_text="Zero Return",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"{category} Daily Returns Distribution (%)",
        xaxis_title="Returns (%)",
        yaxis_title="Frequency",
        template='plotly_white',
        bargap=0.01
    )
    
    return fig

# 헬퍼 함수: 상관관계 차트 생성
def create_correlation_chart(correlations, category):
    """Plotly를 사용한 상관관계 바 차트 생성"""
    import plotly.graph_objects as go
    
    # 상관관계 값에 따라 색상 지정
    colors = ['green' if x >= 0 else 'red' for x in correlations.values]
    
    # 데이터 정렬
    sorted_indices = correlations.abs().argsort().values
    sorted_features = correlations.index[sorted_indices]
    sorted_correlations = correlations.values[sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            y=sorted_features,
            x=sorted_correlations,
            orientation='h',
            marker_color=sorted_colors,
            text=[f"{x:.3f}" for x in sorted_correlations],
            textposition='auto'
        )
    )
    
    fig.update_layout(
        title=f"Top Correlations with {category}",
        xaxis_title="Correlation Coefficient",
        yaxis_title="Indicator",
        template='plotly_white',
        height=500
    )
    
    return fig

# 헬퍼 함수: 상관관계 히트맵 생성
def create_correlation_heatmap(corr_matrix, category):
    """Plotly를 사용한 상관관계 히트맵 생성"""
    import plotly.graph_objects as go
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',  # 빨강-파랑 색상 스케일 (역방향)
        zmid=0,  # 0을 중간값으로 설정
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size":10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        height=600,
        template='plotly_white',
        xaxis={'side': 'top'},
        font=dict( size= 14)  # x축 레이블을 위쪽에 표시
    )
    
    return fig

# 헬퍼 함수: 예측 차트 생성
def create_prediction_chart(recent_predictions, category):
    """Plotly를 사용한 예측 차트 생성"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 실제 가격 변화
    if category in recent_predictions.columns:
        price_changes = recent_predictions[category]
        
        # 색상 설정 (상승/하락에 따라)
        colors = ['green' if x >= 0 else 'red' for x in price_changes]
        
        fig.add_trace(
            go.Bar(
                x=recent_predictions.index,
                y=price_changes,
                name='Actual Price Change',
                marker_color=colors,
                opacity=0.7
            ),
            secondary_y=False
        )
    
    # 예측 결과 (선형 차트)
    if 'predictions' in recent_predictions.columns:
        fig.add_trace(
            go.Scatter(
                x=recent_predictions.index,
                y=recent_predictions['predictions'],
                mode='lines+markers',
                name='Prediction (1=Up, 0=Down)',
                line=dict(color='black', width=2),
                marker=dict(
                    size=8,
                    color=['green' if p == 1 else 'red' for p in recent_predictions['predictions']],
                    line=dict(width=1, color='black')
                )
            ),
            secondary_y=True
        )
    
    # 레이아웃 설정
    fig.update_layout(
        title=f"Recent Predictions for {category}",
        template='plotly_white',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    # 축 레이블 설정
    fig.update_yaxes(title_text="Price Change", secondary_y=False)
    fig.update_yaxes(title_text="Prediction (1=Up, 0=Down)", secondary_y=True,
                    range=[-0.1, 1.1], tickvals=[0, 1], ticktext=['Down', 'Up'])
    fig.update_xaxes(title_text="Date")
    
    return fig
# 헬퍼 함수: 예측 캘린더 시각화
def create_prediction_calendar(calendar_data, category):
    """Plotly를 사용한 예측 캘린더 시각화"""
    import plotly.graph_objects as go
    import pandas as pd
    
    # 데이터 준비
    if not isinstance(calendar_data.index, pd.DatetimeIndex):
        if 'date' in calendar_data.columns:
            calendar_data = calendar_data.set_index('date')
        else:
            return go.Figure()  # 날짜 정보가 없으면 빈 그림 반환
    
    # 날짜를 주 단위로 구성
    calendar_data = calendar_data.copy()
    calendar_data['day_of_week'] = calendar_data.index.dayofweek
    calendar_data['week'] = calendar_data.index.isocalendar().week
    
    # 주별로 그룹화
    weeks = calendar_data['week'].unique()
    
    fig = go.Figure()
    
    # 요일 및 주 레이블
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    week_labels = [f"Week {w}" for w in weeks]
    
    # 각 셀의 색상 및 텍스트 데이터 준비
    text = []
    colors = []
    dates = []
    
    # 각 주에 대해
    for week in weeks:
        week_data = calendar_data[calendar_data['week'] == week]
        
        # 각 요일에 대해
        for day in range(7):
            day_data = week_data[week_data['day_of_week'] == day]
            
            if not day_data.empty:
                date_str = day_data.index[0].strftime('%Y-%m-%d')
                
                # 예측 방향
                if 'predicted_direction' in day_data.columns:
                    pred_dir = day_data['predicted_direction'].iloc[0]
                    color = 'rgba(0, 128, 0, 0.7)' if pred_dir == 'Up' else 'rgba(255, 0, 0, 0.7)'
                    
                    # 실제 방향이 있는 경우
                    if 'actual_direction' in day_data.columns:
                        actual_dir = day_data['actual_direction'].iloc[0]
                        # 예측이 맞았는지 표시
                        if pred_dir == actual_dir:
                            text_content = f"{date_str}\nPred: {pred_dir} ✓"
                        else:
                            text_content = f"{date_str}\nPred: {pred_dir} ✗"
                    else:
                        text_content = f"{date_str}\nPred: {pred_dir}"
                else:
                    text_content = date_str
                    color = 'rgba(200, 200, 200, 0.5)'
            else:
                text_content = ""
                color = 'rgba(255, 255, 255, 0)'
            
            text.append(text_content)
            colors.append(color)
            dates.append(day_data.index[0] if not day_data.empty else None)
    
    # Heatmap 생성
    fig.add_trace(go.Heatmap(
        z=[[1 for _ in range(7)] for _ in range(len(weeks))],  # 더미 데이터
        x=day_labels,
        y=week_labels,
        text=text,
        texttemplate="%{text}",
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],  # 투명 배경
        showscale=False,
        hoverinfo='text'
    ))
    
    # 셀 색상 추가 (개별 사각형으로)
    for i, week in enumerate(weeks):
        for j in range(7):
            idx = i * 7 + j
            if idx < len(colors) and colors[idx] != 'rgba(255, 255, 255, 0)':
                fig.add_shape(
                    type="rect",
                    x0=j-0.45, y0=i-0.45,
                    x1=j+0.45, y1=i+0.45,
                    fillcolor=colors[idx],
                    line=dict(color="white", width=1),
                    layer="below"
                )
    
    fig.update_layout(
        title=f"Prediction Calendar - {category}",
        template='plotly_white',
        height=len(weeks) * 60 + 100,  # 주 수에 따라 높이 조정
        xaxis=dict(
            tickvals=list(range(7)),
            ticktext=day_labels
        ),
        yaxis=dict(
            autorange="reversed",  # 위에서 아래로 주 표시
            tickvals=list(range(len(weeks))),
            ticktext=week_labels
        )
    )
    
    return fig
# 헬퍼 함수: 방향 카운트 차트 생성
def create_direction_counts_chart(labels, actual_counts, predicted_counts):
    """실제 방향 vs 예측 방향 카운트 비교 차트 생성"""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=labels,
            y=actual_counts,
            name='Actual',
            marker_color='royalblue',
            opacity=0.7
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=labels,
            y=predicted_counts,
            name='Predicted',
            marker_color='firebrick',
            opacity=0.7
        )
    )
    
    fig.update_layout(
        title="Actual vs Predicted Direction Counts",
        xaxis_title="Direction",
        yaxis_title="Count",
        template='plotly_white',
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    # 데이터 레이블 추가
    for i, (actual, pred) in enumerate(zip(actual_counts, predicted_counts)):
        fig.add_annotation(
            x=labels[i],
            y=actual,
            text=str(actual),
            showarrow=False,
            yshift=10
        )
        fig.add_annotation(
            x=labels[i],
            y=pred,
            text=str(pred),
            showarrow=False,
            yshift=10,
            xshift=20
        )
    
    return fig
# 헬퍼 함수: 혼동 행렬 차트 생성
def create_confusion_matrix_chart(cm, category):
    """혼동 행렬 히트맵 생성"""
    import plotly.graph_objects as go
    import numpy as np
    
    # 백분율로 변환
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 레이블 및 주석 텍스트 준비
    labels = ['Down', 'Up']
    annotations = []
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=f"{cm[i, j]} ({cm_percent[i, j]:.1f}%)",
                    font=dict(color='white' if cm[i, j] > cm.max() / 2 else 'black'),
                    showarrow=False
                )
            )
    
    # 색상 스케일 설정
    colorscale = [
        [0, 'rgb(247, 251, 255)'],
        [0.1, 'rgb(222, 235, 247)'],
        [0.2, 'rgb(198, 219, 239)'],
        [0.3, 'rgb(158, 202, 225)'],
        [0.4, 'rgb(107, 174, 214)'],
        [0.5, 'rgb(66, 146, 198)'],
        [0.6, 'rgb(33, 113, 181)'],
        [0.7, 'rgb(8, 81, 156)'],
        [0.8, 'rgb(8, 48, 107)'],
        [1, 'rgb(3, 19, 43)']
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=colorscale,
        text=[[f"{cm[i, j]} ({cm_percent[i, j]:.1f}%)" for j in range(len(labels))] for i in range(len(labels))],
        texttemplate="%{text}",
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f"Confusion Matrix - {category}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        annotations=annotations,
        template='plotly_white'
    )
    
    return fig

# 헬퍼 함수: 정확도 추이 차트 생성
def create_accuracy_trend_chart(rolling_accuracy, window_size, category):
    """시간에 따른 예측 정확도 추이 차트 생성"""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=rolling_accuracy.index,
            y=rolling_accuracy.values,
            mode='lines',
            name=f'{window_size}-period Rolling Accuracy',
            line=dict(color='royalblue', width=2)
        )
    )
    
    # 50% 라인 추가 (랜덤 예측)
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Random Guess (50%)",
        annotation_position="bottom right"
    )
    
    # 전체 평균 정확도 라인 추가
    mean_accuracy = rolling_accuracy.mean()
    fig.add_hline(
        y=mean_accuracy,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Avg Accuracy: {mean_accuracy:.2%}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"Prediction Accuracy Trend - {category}",
        xaxis_title="Date",
        yaxis_title="Accuracy (Rolling Window)",
        template='plotly_white',
        yaxis=dict(tickformat='.0%', range=[0, 1])
    )
    
    return fig

# 헬퍼 함수: 예측 캘린더 시각화
def create_prediction_calendar(calendar_data, category):
    """Plotly를 사용한 예측 캘린더 시각화"""
    import plotly.graph_objects as go
    import pandas as pd
    
    # 데이터 준비
    if not isinstance(calendar_data.index, pd.DatetimeIndex):
        if 'date' in calendar_data.columns:
            calendar_data = calendar_data.set_index('date')
        else:
            return go.Figure()  # 날짜 정보가 없으면 빈 그림 반환
    
    # 날짜를 주 단위로 구성
    calendar_data = calendar_data.copy()
    calendar_data['day_of_week'] = calendar_data.index.dayofweek
    calendar_data['week'] = calendar_data.index.isocalendar().week
    
    # 주별로 그룹화
    weeks = calendar_data['week'].unique()
    
    fig = go.Figure()
    
    # 요일 및 주 레이블
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    week_labels = [f"Week {w}" for w in weeks]
    
    # 각 셀의 색상 및 텍스트 데이터 준비
    text = []
    colors = []
    dates = []
    
    # 각 주에 대해
    for week in weeks:
        week_data = calendar_data[calendar_data['week'] == week]
        
        # 각 요일에 대해
        for day in range(7):
            day_data = week_data[week_data['day_of_week'] == day]
            
            if not day_data.empty:
                date_str = day_data.index[0].strftime('%Y-%m-%d')
                
                # 예측 방향
                if 'predicted_direction' in day_data.columns:
                    pred_dir = day_data['predicted_direction'].iloc[0]
                    color = 'rgba(0, 128, 0, 0.7)' if pred_dir == 'Up' else 'rgba(255, 0, 0, 0.7)'
                    
                    # 실제 방향이 있는 경우
                    if 'actual_direction' in day_data.columns:
                        actual_dir = day_data['actual_direction'].iloc[0]
                        # 예측이 맞았는지 표시
                        if pred_dir == actual_dir:
                            text_content = f"{date_str}\nPred: {pred_dir} ✓"
                        else:
                            text_content = f"{date_str}\nPred: {pred_dir} ✗"
                    else:
                        text_content = f"{date_str}\nPred: {pred_dir}"
                else:
                    text_content = date_str
                    color = 'rgba(200, 200, 200, 0.5)'
            else:
                text_content = ""
                color = 'rgba(255, 255, 255, 0)'
            
            text.append(text_content)
            colors.append(color)
            dates.append(day_data.index[0] if not day_data.empty else None)
    
    # Heatmap 생성
    fig.add_trace(go.Heatmap(
        z=[[1 for _ in range(7)] for _ in range(len(weeks))],  # 더미 데이터
        x=day_labels,
        y=week_labels,
        text=text,
        texttemplate="%{text}",
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],  # 투명 배경
        showscale=False,
        hoverinfo='text'
    ))
    
    # 셀 색상 추가 (개별 사각형으로)
    for i, week in enumerate(weeks):
        for j in range(7):
            idx = i * 7 + j
            if idx < len(colors) and colors[idx] != 'rgba(255, 255, 255, 0)':
                fig.add_shape(
                    type="rect",
                    x0=j-0.45, y0=i-0.45,
                    x1=j+0.45, y1=i+0.45,
                    fillcolor=colors[idx],
                    line=dict(color="white", width=1),
                    layer="below"
                )
    
    fig.update_layout(
    title=f"Prediction Calendar - {category}",
    template='plotly_white',
    height=len(weeks) * 60 + 100,  # 주 수에 따라 높이 조정
    xaxis=dict(
        tickvals=list(range(7)),
        ticktext=day_labels
    ),
    yaxis=dict(
        autorange="reversed",  # 위에서 아래로 주 표시
        tickvals=list(range(len(weeks))),
        ticktext=week_labels
    )
)

    return fig

def create_asset_comparison_chart(daily_prices, assets, start_date=None, end_date=None):
    """여러 자산의 성과를 비교하는 차트 생성"""
    import plotly.graph_objects as go
    import pandas as pd
    
    fig = go.Figure()
    
    # 날짜 필터링
    if start_date is not None and end_date is not None:
        mask = (daily_prices.index.date >= start_date) & (daily_prices.index.date <= end_date)
        filtered_prices = daily_prices.loc[mask]
    else:
        filtered_prices = daily_prices
    
    # 비교를 위해 첫 날을 100으로 정규화
    normalized_prices = pd.DataFrame()
    
    for asset in assets:
        if asset in filtered_prices.columns:
            asset_prices = filtered_prices[asset].dropna()
            if not asset_prices.empty:
                normalized_prices[asset] = asset_prices / asset_prices.iloc[0] * 100
    
    # 각 자산의 성과 그래프 추가
    for asset in assets:
        if asset in normalized_prices.columns:
            color = 'royalblue' if asset == assets[0] else None  # 첫 번째 자산(선택된 자산)은 파란색으로 강조
            width = 3 if asset == assets[0] else 1.5  # 첫 번째 자산은 더 두껍게
            
            fig.add_trace(
                go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices[asset],
                    mode='lines',
                    name=asset,
                    line=dict(color=color, width=width)
                )
            )
    
    # 기준선 (100) 추가
    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="black",
        annotation_text="Baseline (100)",
        annotation_position="bottom right"
    )
    
    # 레이아웃 설정
    fig.update_layout(
        title="Asset Performance Comparison (Normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Price (First day = 100)",
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return fig
def create_performance_comparison_chart(performance_df):
    """모델 성능 비교 차트 생성"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    if performance_df.empty:
        # 빈 차트 반환
        return go.Figure()
    
    # 데이터 준비
    categories = performance_df['Category'].tolist()
    accuracy = performance_df['Accuracy'].tolist()
    optimal_accuracy = performance_df['Optimal Threshold Accuracy'].tolist()
    roc_auc = performance_df['ROC-AUC'].tolist()
    
    # 차트 생성
    fig = make_subplots(rows=1, cols=1)
    
    # 정확도 바
    fig.add_trace(
        go.Bar(
            x=categories,
            y=accuracy,
            name='Accuracy',
            marker_color='royalblue',
            opacity=0.7
        )
    )
    
    # 최적 임계값 정확도 바
    fig.add_trace(
        go.Bar(
            x=categories,
            y=optimal_accuracy,
            name='Optimal Accuracy',
            marker_color='firebrick',
            opacity=0.7
        )
    )
    
    # ROC-AUC 선
    fig.add_trace(
        go.Scatter(
            x=categories,
            y=roc_auc,
            mode='lines+markers',
            name='ROC-AUC',
            line=dict(color='green', width=2),
            marker=dict(size=10)
        )
    )
    
    # 레이아웃 설정
    fig.update_layout(
        title="Model Performance Metrics by Category",
        xaxis_title="Category",
        yaxis_title="Score",
        template='plotly_white',
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        yaxis=dict(tickformat='.0%', range=[0, 1])
    )
    
    return fig
import streamlit as st
import pandas as pd

# --- 여기에 필요한 함수들 import (또는 정의) 해줘야 해 ---
# preprocess_data, create_correlation_features, get_indicator_columns,
# prepare_enhanced_model_data, build_improved_prediction_model,
# generate_focused_ml_insights, generate_market_insights_dashboard,
# extreme_events_dashboard, create_output_directory

def main():
    """Streamlit 메인 실행 함수"""

    # 세션 상태 초기화
    if 'category_analyses' not in st.session_state:
        st.session_state['category_analyses'] = {}
    if 'current_category' not in st.session_state:
        st.session_state['current_category'] = None

    # 페이지 타이틀
    st.title("Financial Data Analysis & Prediction App")

    # 사이드바 설정
    st.sidebar.header("Setting")
    st.sidebar.subheader("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

    app_mode = st.sidebar.selectbox("Select Mode", ["🤖 ML: Stock Up/Down Prediction", "🚨 Extreme Events & Anomalies"])

    st.sidebar.subheader(" Model Settings")
    n_features = st.sidebar.slider("Number of Features to Select", 10, 70, 30)
    n_trials = st.sidebar.slider("Number of Optuna Trials", 10, 150, 50)

    run_analysis = st.sidebar.button("🚀Run Analysis")

    # 데이터 존재 여부 체크
    if uploaded_file is not None or (
        'df' in st.session_state and
        'daily_avg_prices' in st.session_state and
        'daily_indicators' in st.session_state and
        'enhanced_model_data' in st.session_state
    ):

        # 파일 업로드 시 데이터 처리
        if uploaded_file is not None:
            with st.spinner("Loading data..."):
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state['df'] = df
                    st.subheader(" Data Preview")
                    st.dataframe(df.head())

                    st.subheader("Basic Info")
                    st.write(f"- **Rows**: {df.shape[0]}")
                    st.write(f"- **Columns**: {df.shape[1]}")

                    required_cols = ['date', 'category', 'close']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        st.error(f" Missing required columns: {missing_cols}")
                        st.stop()

                    if 'date' in df.columns:
                        if not pd.api.types.is_datetime64_any_dtype(df['date']):
                            df['date'] = pd.to_datetime(df['date'], errors='coerce')

                    if 'category' in df.columns:
                        categories_in_data = df['category'].unique().tolist()
                        st.write(f"Categories: {categories_in_data}")
                        categories = st.sidebar.multiselect(
                            "Select Categories to Analyze",
                            categories_in_data,
                            default=categories_in_data[:min(4, len(categories_in_data))]
                        )
                        st.session_state['categories'] = categories
                    else:
                        st.error("'category' column is missing in the data.")
                        st.stop()

                    with st.spinner("Preprocessing data..."):
                        df, daily_avg_prices, daily_indicators = preprocess_data(df)

                except Exception as e:
                    st.error(f" Error during data processing: {str(e)}")
                    st.exception(e)
                    return

        # 분석 모드 분기
        if app_mode == "🤖 ML: Stock Up/Down Prediction":
            model_tab, insight_tab = st.tabs(["📊 Model Result", "🧠 Market Insight"])

            if run_analysis and uploaded_file is not None:
                output_dir = create_output_directory("ML_results")

                with st.spinner("Creating correlation features..."):
                    category_tickers = {cat: [cat] for cat in categories}
                    df_corr, daily_avg_prices, daily_indicators = create_correlation_features(df, category_tickers)

                _, _, _, all_norm_indicators = get_indicator_columns()
                important_indicators = [ind for ind in all_norm_indicators if ind in daily_indicators.columns]

                enhanced_model_data = prepare_enhanced_model_data(
                    daily_avg_prices, daily_indicators, categories, important_indicators
                )

                results = {}
                for category in categories:
                    if category in enhanced_model_data:
                        result = build_improved_prediction_model(
                            enhanced_model_data, category, output_dir,
                            n_features=n_features, n_trials=n_trials
                        )
                        results[category] = result

                # 세션에 결과 저장
                st.session_state['model_results'] = results
                st.session_state['enhanced_model_data'] = enhanced_model_data
                st.session_state['daily_avg_prices'] = daily_avg_prices
                st.session_state['daily_indicators'] = daily_indicators
                st.session_state['df'] = df

                st.success("The analysis is complete. Click each tab to view the results.")

                with model_tab:
                    generate_focused_ml_insights(
                        model_results=results,
                        model_data=enhanced_model_data,
                        daily_avg_prices=daily_avg_prices,
                        daily_indicators=daily_indicators
                    )
                with insight_tab:
                    generate_market_insights_dashboard(
                        enhanced_model_data=enhanced_model_data,
                        daily_avg_prices=daily_avg_prices,
                        daily_indicators=daily_indicators
                    )
            else:
                # 분석 없이 세션 데이터만으로 대시보드 표시
                if 'enhanced_model_data' in st.session_state:
                    enhanced_model_data = st.session_state['enhanced_model_data']
                    results = st.session_state['model_results']
                    daily_avg_prices = st.session_state['daily_avg_prices']
                    daily_indicators = st.session_state['daily_indicators']

                    with model_tab:
                        generate_focused_ml_insights(
                            model_results=results,
                            model_data=enhanced_model_data,
                            daily_avg_prices=daily_avg_prices,
                            daily_indicators=daily_indicators
                        )
                    with insight_tab:
                        generate_market_insights_dashboard(
                            enhanced_model_data=enhanced_model_data,
                            daily_avg_prices=daily_avg_prices,
                            daily_indicators=daily_indicators
                        )
                else:
                    st.info(" Please run the analysis first.")

        elif app_mode == "🚨 Extreme Events & Anomalies":
            required_keys = ['enhanced_model_data', 'df', 'daily_avg_prices', 'daily_indicators']
            missing_keys = [key for key in required_keys if key not in st.session_state]

            if missing_keys:
                st.error(f" Missing data: {', '.join(missing_keys)}. Please run analysis first in App1.")
                st.stop()

            enhanced_model_data = st.session_state['enhanced_model_data']
            df = st.session_state['df']
            daily_avg_prices = st.session_state['daily_avg_prices']
            daily_indicators = st.session_state['daily_indicators']
            categories = st.session_state['categories']

            extreme_events_dashboard(df, daily_avg_prices, daily_indicators)

    else:
        st.info(" Please upload a CSV file from the sidebar to get started.")


if __name__ == "__main__":
    main()

# def main():
#     """Streamlit 메인 실행 함수"""

#     # 세션 상태 초기화
#     if 'category_analyses' not in st.session_state:
#         st.session_state['category_analyses'] = {}
#     if 'current_category' not in st.session_state:
#         st.session_state['current_category'] = None

#     # 페이지 타이틀
#     st.title(" Financial Data Analysis & Prediction App")

#     # 사이드바 설정
#     st.sidebar.header(" Setting")
#     st.sidebar.subheader("Data Upload")
#     uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

#     app_mode = st.sidebar.selectbox("Select Mode", ["🤖 ML: Stock Up/Down Prediction", "🚨 Extreme Events & Anomalies"])

#     st.sidebar.subheader(" Model Settings")
#     n_features = st.sidebar.slider(" Number of Features to Select", 10, 70, 30)
#     n_trials = st.sidebar.slider(" Number of Optuna Trials", 10, 150, 50)

#     run_analysis = st.sidebar.button(" Run Analysis")

#     if uploaded_file is not None or ('df' in st.session_state and 'daily_avg_prices' in st.session_state and 'daily_indicators' in st.session_state and 'enhanced_model_data' in st.session_state):
#         if uploaded_file is not None:
#             with st.spinner("Loading data..."):
#                 try:
#                     df = pd.read_csv(uploaded_file)
#                     st.session_state['df'] = df
#                     st.subheader("Data Preview")
#                     st.dataframe(df.head())

#                     st.subheader("Basic Info")
#                     st.write(f"- **Rows**: {df.shape[0]}")
#                     st.write(f"- **Columns**: {df.shape[1]}")

#                     required_cols = ['date', 'category', 'close']
#                     missing_cols = [col for col in required_cols if col not in df.columns]
#                     if missing_cols:
#                         st.error(f"Missing required columns: {missing_cols}")
#                         st.stop()

#                     if 'date' in df.columns:
#                         if not pd.api.types.is_datetime64_any_dtype(df['date']):
#                             df['date'] = pd.to_datetime(df['date'], errors='coerce')

#                     if 'category' in df.columns:
#                         categories_in_data = df['category'].unique().tolist()
#                         st.write(f"\ Categories: {categories_in_data}")
#                         categories = st.sidebar.multiselect(
#                             "Select Categories to Analyze",
#                             categories_in_data,
#                             default=categories_in_data[:min(4, len(categories_in_data))]
#                         )
#                     else:
#                         st.error("'category' column is missing in the data.")
#                         st.stop()

#                     with st.spinner("Preprocessing data..."):
#                         df, daily_avg_prices, daily_indicators = preprocess_data(df)
#                         st.session_state['daily_avg_prices'] = daily_avg_prices
#                         st.session_state['daily_indicators'] = daily_indicators

#                 except Exception as e:
#                     st.error(f" Error during data processing: {str(e)}")
#                     st.exception(e)
#                     return

#         # 분석 모드 분기
#         if app_mode == "🤖 ML: Stock Up/Down Prediction":
#             model_tab, insight_tab = st.tabs(["📊 Model Result", "🧠 Market Insight"])

#             if run_analysis and uploaded_file is not None:
#                 output_dir = create_output_directory("ML_results")

#                 with st.spinner("Creating correlation features..."):
#                     category_tickers = {cat: [cat] for cat in categories}
#                     df_corr, daily_avg_prices, daily_indicators = create_correlation_features(df, category_tickers)

#                 # ✨ important_indicators 정의 추가
#                 _, _, _, all_norm_indicators = get_indicator_columns()
#                 important_indicators = [ind for ind in all_norm_indicators if ind in daily_indicators.columns]

#                 enhanced_model_data = prepare_enhanced_model_data(
#                     daily_avg_prices, daily_indicators, categories, important_indicators
#                 )

#                 results = {}
#                 for category in categories:
#                     if category in enhanced_model_data:
#                         result = build_improved_prediction_model(
#                             enhanced_model_data, category, output_dir,
#                             n_features=n_features, n_trials=n_trials
#                         )
#                         results[category] = result

#                 st.session_state['model_results'] = results
#                 st.session_state['enhanced_model_data'] = enhanced_model_data
#                 st.session_state['daily_avg_prices'] = daily_avg_prices
#                 st.session_state['daily_indicators'] = daily_indicators
#                 st.session_state['df'] = df
#                 st.success("✅ The analysis is complete. Click each tab to view the results.")

#                 with model_tab:
#                     generate_focused_ml_insights(
#                         model_results=results,
#                         model_data=enhanced_model_data,
#                         daily_avg_prices=daily_avg_prices,
#                         daily_indicators=daily_indicators
#                     )
#                 with insight_tab:
#                     generate_market_insights_dashboard(
#                         enhanced_model_data=enhanced_model_data,
#                         daily_avg_prices=daily_avg_prices,
#                         daily_indicators=daily_indicators
#                     )

#             else:
#                 if 'enhanced_model_data' in st.session_state:
#                     enhanced_model_data = st.session_state['enhanced_model_data']
#                     results = st.session_state['model_results']
#                     daily_avg_prices = st.session_state['daily_avg_prices']
#                     daily_indicators = st.session_state['daily_indicators']
                    
#                     with model_tab:
#                         generate_focused_ml_insights(
#                             model_results=results,
#                             model_data=enhanced_model_data,
#                             daily_avg_prices=daily_avg_prices,
#                             daily_indicators=daily_indicators
#                         )
#                     with insight_tab:
#                         generate_market_insights_dashboard(
#                             enhanced_model_data=enhanced_model_data,
#                             daily_avg_prices=daily_avg_prices,
#                             daily_indicators=daily_indicators
#                         )
#                 else:
#                     st.info("Please run the analysis first.")

#         elif app_mode == "🚨 Extreme Events & Anomalies":
#             required_keys = ['enhanced_model_data', 'df', 'daily_avg_prices', 'daily_indicators']
#             missing_keys = [key for key in required_keys if key not in st.session_state]

#             if missing_keys:
#                 st.error(f"❌ Missing data: {', '.join(missing_keys)}. Please run analysis first in App1.")
#                 st.stop()

#             # ✅ 세션에서 데이터 가져오기
#             enhanced_model_data = st.session_state['enhanced_model_data']
#             df = st.session_state['df']
#             daily_avg_prices = st.session_state['daily_avg_prices']
#             daily_indicators = st.session_state['daily_indicators']

#             # ✅ 대시보드 호출
#             extreme_events_dashboard(df, daily_avg_prices, daily_indicators)

#     else:
#         st.info(" Please upload a CSV file from the sidebar to get started.")

# if __name__ == "__main__":
#     main()


