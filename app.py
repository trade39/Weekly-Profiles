import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, time
import pytz
import requests
from bs4 import BeautifulSoup
import re
import cot_reports as cot  # NEW IMPORT

# ML Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ICT Profiles Analyzer + ML",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# --- CUSTOM CSS FOR DEEP NAVY / CYAN MONOCHROMATIC LOOK ---
st.markdown("""
<style>
    /* 1. Primary Background: Deep Navy */
    [data-testid="stAppViewContainer"] {
        background-color: #0A0F1E;
        color: #FFFFFF;
    }
    
    /* 2. Sidebar Background: Slightly lighter dark gray-blue */
    [data-testid="stSidebar"] {
        background-color: #12161F;
        border-right: 1px solid #1E252F;
    }
    
    /* 3. Metric/Card Backgrounds */
    div[data-testid="metric-container"] {
        background-color: #1E252F;
        border: 1px solid #2B3442;
        padding: 15px;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    [data-testid="stMetricLabel"] {
        color: #CCCCCC; /* Secondary Label Color */
        font-size: 14px;
    }
    
    [data-testid="stMetricValue"] {
        color: #FFFFFF; /* Primary Data Color */
        font-weight: 600;
    }
    
    /* 4. Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #333333;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #0A0F1E;
        color: #CCCCCC;
        border: 1px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1E252F;
        color: #00FFFF; /* Cyan Accent */
        border-bottom: 2px solid #00FFFF;
    }
    
    /* 5. Inputs and Selectboxes */
    .stSelectbox div[data-baseweb="select"] div {
        background-color: #1E252F;
        color: white;
        border-color: #333333;
    }
    
    .stDateInput div[data-baseweb="input"] {
        background-color: #1E252F;
        color: white;
        border-color: #333333;
    }

    .stNumberInput div[data-baseweb="input"] {
        background-color: #1E252F;
        color: white;
        border-color: #333333;
    }
    
    /* Global Text Adjustments */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    p, li, span, label {
        color: #CCCCCC;
    }
    
    /* Divider Color */
    hr {
        border-color: #333333;
    }
    
    /* Table Styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #333333;
    }
</style>
""", unsafe_allow_html=True)

# --- THEME CONFIGURATION FOR PLOTLY ---
THEME = {
    "background": "#0A0F1E",
    "paper_bg": "#0A0F1E", 
    "grid": "#333333",
    "text": "#FFFFFF",
    "bullish": "#00FFFF",  # Cyan (Growth/Up)
    "bearish": "#8080FF",  # Mid-tone Blue (Down)
    "accent_fill": "rgba(102, 204, 255, 0.2)",
    "line_primary": "#00FFFF",
    "line_secondary": "#8080FF"
}

def apply_theme(fig):
    """Applies the monochromatic blue/cyan theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME['paper_bg'],
        plot_bgcolor=THEME['background'],
        font=dict(color=THEME['text']),
        xaxis=dict(gridcolor=THEME['grid'], showgrid=True, zerolinecolor=THEME['grid']),
        yaxis=dict(gridcolor=THEME['grid'], showgrid=True, zerolinecolor=THEME['grid']),
    )
    return fig

# --- SIDEBAR SETTINGS ---
st.sidebar.title("⚙️ Configuration")

# 1. Analysis Mode
analysis_mode = st.sidebar.radio(
    "Analysis Mode", 
    [
        "Weekly Protocols (News & Logic)", 
        "Weekly Profiles", 
        "Intraday Profiles", 
        "One Shot One Kill (OSOK)",
        "COT Quant Terminal"  # NEW MODE
    ], 
    help="Select the analytical framework."
)

st.sidebar.markdown("---")

# 2. Asset Selection Logic (Conditional based on Mode)
if analysis_mode == "COT Quant Terminal":
    # COT CONFIGURATION
    COT_ASSET_CONFIG = {
        "Gold (GC)": {
            "keywords": ["GOLD", "COMMODITY EXCHANGE"],
            "report_type": "disaggregated_fut",
            "labels": ("Managed Money", "Producer/Merchant/Processor/User") 
        },
        "S&P 500 (ES)": {
            "keywords": ["E-MINI S&P 500", "CHICAGO MERCANTILE EXCHANGE"],
            "report_type": "traders_in_financial_futures_fut",
            "labels": ("Leveraged Funds", "Dealer/Intermediary") 
        },
        "Nasdaq 100 (NQ)": {
            "keywords": ["E-MINI NASDAQ-100", "CHICAGO MERCANTILE EXCHANGE"],
            "report_type": "traders_in_financial_futures_fut",
            "labels": ("Leveraged Funds", "Dealer/Intermediary")
        },
        "Japanese Yen (6J)": {
            "keywords": ["JAPANESE YEN", "CHICAGO MERCANTILE EXCHANGE"],
            "report_type": "traders_in_financial_futures_fut",
            "labels": ("Leveraged Funds", "Asset Manager/Institutional")
        },
        "EUR/USD (6E)": {
            "keywords": ["EURO FX", "CHICAGO MERCANTILE EXCHANGE"],
            "report_type": "traders_in_financial_futures_fut",
            "labels": ("Leveraged Funds", "Asset Manager/Institutional")
        },
        "Bitcoin (BTC)": {
            "keywords": ["BITCOIN", "CHICAGO MERCANTILE EXCHANGE"],
            "report_type": "traders_in_financial_futures_fut",
            "labels": ("Leveraged Funds", "Asset Manager/Institutional")
        }
    }
    
    selected_cot_asset = st.sidebar.selectbox("Select Asset (COT)", list(COT_ASSET_CONFIG.keys()))
    current_year = datetime.now().year
    cot_start_year = st.sidebar.number_input("Lookback Start Year", min_value=2015, max_value=current_year, value=2020)
    
else:
    # STANDARD CONFIGURATION
    # 2. Fallback Toggle
    use_etf = st.sidebar.checkbox("Use ETF Tickers (More Stable)", value=False, 
        help="Check this if Futures data (ES=F, GC=F) fails to load.")

    # 3. Asset Selection
    if use_etf:
        asset_map = {
            "Gold (GLD ETF)": "GLD",
            "S&P 500 (SPY ETF)": "SPY",
            "Nasdaq 100 (QQQ ETF)": "QQQ",
            "EUR/USD": "EURUSD=X",
            "GBP/USD": "GBPUSD=X",
            "Bitcoin": "BTC-USD"
        }
    else:
        asset_map = {
            "Gold (Futures)": "GC=F",
            "S&P 500 (E-mini Futures)": "ES=F",
            "Nasdaq 100 (E-mini Futures)": "NQ=F",
            "EUR/USD": "EURUSD=X",
            "GBP/USD": "GBPUSD=X",
            "Bitcoin": "BTC-USD"
        }

    selected_asset_name = st.sidebar.selectbox("Select Asset", list(asset_map.keys()))
    ticker_symbol = asset_map[selected_asset_name]


# --- SCRAPING & NEWS FUNCTIONS ---

@st.cache_data(ttl=3600)
def scrape_forex_factory(week="this"):
    """
    Scrapes Forex Factory Calendar for High Impact (Red) News.
    """
    url = f"https://www.forexfactory.com/calendar?week={week}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return pd.DataFrame()
            
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find("table", class_="calendar__table")
        
        if not table:
            return pd.DataFrame()

        data = []
        rows = table.find_all("tr", class_=lambda x: x and "calendar__row" in x and "calendar_row" in x)
        
        current_date = None
        
        for row in rows:
            # Extract Date (merged cell handling)
            date_cell = row.find("td", class_="calendar__date")
            if date_cell and date_cell.text.strip():
                current_date = date_cell.text.strip()
            
            # Extract Currency & Impact
            currency = row.find("td", class_="calendar__currency").text.strip()
            impact_cell = row.find("td", class_="calendar__impact")
            impact_span = impact_cell.find("span") if impact_cell else None
            impact_class = impact_span.get("class", []) if impact_span else []
            
            # Identify Impact Color
            impact = "Low"
            if "icon--impact-red" in impact_class: impact = "High"
            elif "icon--impact-orange" in impact_class: impact = "Medium"
            elif "icon--impact-yellow" in impact_class: impact = "Low"
            
            event_name = row.find("td", class_="calendar__event").text.strip()
            time_str = row.find("td", class_="calendar__time").text.strip()
            
            if impact == "High": # Filter for Red Folder only
                data.append({
                    "Date": current_date,
                    "Time": time_str,
                    "Currency": currency,
                    "Event": event_name,
                    "Impact": impact
                })
                
        return pd.DataFrame(data)

    except Exception as e:
        # Fail gracefully
        return pd.DataFrame()

def get_financial_news():
    """Fetches latest news for EURUSD/USD to detect Exogenous Events."""
    try:
        ticker = yf.Ticker("EURUSD=X") 
        news = ticker.news
        stories = []
        for n in news[:5]:
            stories.append({
                "Title": n.get('title', "No Title"),
                "Publisher": n.get('publisher', "Unknown"),
                "Link": n.get('link', "#"),
                "Time": datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M')
            })
        return stories
    except:
        return []

def determine_weekly_protocol(ff_df, stories, manual_override_date=None):
    """
    Implements the ICT Decision Tree for Weekly Profiles.
    """
    # 1. Setup Context
    today = manual_override_date if manual_override_date else datetime.now().date()
    day_name = today.strftime("%A")
    
    # 2. Analyze News Clustering (USD High Impact)
    if ff_df is not None and not ff_df.empty:
        usd_news = ff_df[ff_df['Currency'] == 'USD']
    else:
        usd_news = pd.DataFrame()
    
    clustering = "Minimal/None"
    red_folder_days = []
    
    if not usd_news.empty:
        # Simple parsing of day names from the Date string
        usd_news['DayName'] = usd_news['Date'].apply(lambda x: x.split(' ')[0] if x else "")
        red_folder_days = usd_news['DayName'].unique().tolist()
        
        has_early = any(d in ['Mon', 'Tue'] for d in red_folder_days)
        has_mid = any(d in ['Wed', 'Thu'] for d in red_folder_days)
        has_late = any(d in ['Thu', 'Fri'] for d in red_folder_days)
        
        if has_early and not has_mid: clustering = "Early Week (Mon-Tue)"
        elif has_mid and not has_early: clustering = "Midweek (Tue-Thu)"
        elif has_late and not has_mid: clustering = "Late Week (Thu-Fri)"
        elif has_mid: clustering = "Midweek (Tue-Thu)" 
    
    # 3. Detect Exogenous Events (Keywords in Headlines)
    keywords = ["War", "Sanctions", "Invasion", "Emergency", "Crisis", "Attack", "OPEC", "Rate Hike"]
    exogenous_found = False
    exo_headline = ""
    
    if stories:
        for s in stories:
            title = s.get('Title', "")
            if not isinstance(title, str): continue
            
            if any(k.lower() in title.lower() for k in keywords):
                exogenous_found = True
                exo_headline = title
                break
            
    # 4. Decision Tree Logic
    protocol = {}
    
    # Default Rule: Avoid Monday
    if day_name == "Monday":
        if exogenous_found:
            protocol = {
                "Action": "⚠️ ALERT: Monday Exception Active",
                "Profile": "Seek and Destroy / Early Expansion",
                "Logic": f"Catalyst Detected: '{exo_headline}'. Volatility injection overrides default avoidance.",
                "Bias": "Follow Displacement from H4 Arrays."
            }
        else:
            protocol = {
                "Action": "⛔ WAIT: Avoid Monday (Default)",
                "Profile": "Data Insufficient / Consolidation",
                "Logic": "No high-impact scheduled news & no exogenous shocks. Expect manipulation/range.",
                "Bias": "Neutral / Internal Range Equilibrium."
            }
    else:
        # Mid-Week Logic based on Clustering
        if clustering == "Minimal/None":
            protocol = {
                "Action": "Trade Internal Range / Scalp",
                "Profile": "Continued Consolidation",
                "Logic": "Lack of scheduled volatility drivers.",
                "Bias": "Range Bound (Fade Highs/Lows)."
            }
        elif clustering == "Early Week (Mon-Tue)":
            protocol = {
                "Action": "Look for Tuesday Reversal",
                "Profile": "Early Week Expansion",
                "Logic": "News loaded early; Mon/Tue forms the low/high of week.",
                "Bias": "Expansion Wed-Fri."
            }
        elif clustering == "Midweek (Tue-Thu)":
            protocol = {
                "Action": "Wait for Midweek Manipulation",
                "Profile": "Classic Expansion / Midweek Reversal",
                "Logic": "Tue/Wed news acts as the Judas Swing.",
                "Bias": "Reversal likely Wed; Trade Thu Expansion."
            }
        elif clustering == "Late Week (Thu-Fri)":
            protocol = {
                "Action": "Patience until Thursday",
                "Profile": "Consolidation Reversal",
                "Logic": "Early week trap; Real move comes late.",
                "Bias": "Thu/Fri Reversal."
            }
            
    return protocol, clustering, red_folder_days, exogenous_found

# --- SHARED HELPER FUNCTIONS (YFINANCE) ---

def get_data_weekly(ticker, weeks=52):
    """Fetches daily data for Weekly Analysis."""
    min_history_weeks = 300 
    fetch_weeks = max(weeks, min_history_weeks)
    period_days = fetch_weeks * 7 + 30
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        # Adjust to start of week (Monday)
        data['Week_Start'] = data['Date'].apply(lambda x: x - timedelta(days=x.weekday()))
        return data
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def get_data_intraday(ticker, target_date, interval="5m"):
    """
    Fetches intraday data. Interval can be 5m (Intraday) or 15m (OSOK).
    """
    start_date = target_date - timedelta(days=2) 
    end_date = target_date + timedelta(days=2)
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if data.empty: return None
        
        # Flatten MultiIndex
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        
        # Robust Timezone Logic
        if data['Datetime'].dt.tz is None:
            data['Datetime'] = data['Datetime'].dt.tz_localize('UTC')
        else:
            data['Datetime'] = data['Datetime'].dt.tz_convert('UTC')
        
        # Convert to NY for ICT Analysis
        data['Datetime_NY'] = data['Datetime'].dt.tz_convert('America/New_York')
        
        # Strict Date Filtering on NY Time
        mask = data['Datetime_NY'].dt.date == target_date
        return data.loc[mask].copy()
    except Exception as e:
        st.error(f"Error fetching intraday data: {e}")
        return None

# --- SHARED HELPER FUNCTIONS (COT DATA) ---

def clean_headers(df):
    if isinstance(df.columns[0], int):
        for i in range(20):
            row_str = " ".join(df.iloc[i].astype(str).tolist()).lower()
            if "market" in row_str and ("long" in row_str or "positions" in row_str):
                df.columns = df.iloc[i]
                return df.iloc[i+1:].reset_index(drop=True)
    return df

def map_columns(df, report_type):
    col_map = {}
    
    def get_col(keywords, exclude=None):
        for col in df.columns:
            c_str = str(col).lower()
            if all(k in c_str for k in keywords):
                if exclude and any(x in c_str for x in exclude): continue
                return col
        return None

    col_map['date'] = get_col(['report', 'date']) or get_col(['as', 'of', 'date']) or get_col(['date'])
    col_map['market'] = get_col(['market'])

    # Logic to capture "Commercial" equivalent based on report type
    if "disaggregated" in report_type:
        col_map['spec_long'] = get_col(['money', 'long'], exclude=['lev'])
        col_map['spec_short'] = get_col(['money', 'short'], exclude=['lev'])
        # "Commercials" are Producers/Merchants
        col_map['hedge_long'] = get_col(['prod', 'merc', 'long'])
        col_map['hedge_short'] = get_col(['prod', 'merc', 'short'])
        
    elif "financial" in report_type:
        col_map['spec_long'] = get_col(['lev', 'money', 'long'])
        col_map['spec_short'] = get_col(['lev', 'money', 'short'])
        # "Commercials" approximation in Financials (Dealers/Asset Managers)
        col_map['hedge_long'] = get_col(['dealer', 'long']) or get_col(['asset', 'long'])
        col_map['hedge_short'] = get_col(['dealer', 'short']) or get_col(['asset', 'short'])

    final_map = {v: k for k, v in col_map.items() if v}
    df = df.rename(columns=final_map)
    
    for c in ['spec_long', 'spec_short', 'hedge_long', 'hedge_short']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
    return df

@st.cache_data(ttl=3600)
def load_cot_data(year, report_type):
    try:
        df = cot.cot_year(year=year, cot_report_type=report_type)
        if df is None or df.empty: return None, "Empty File"
        
        df = clean_headers(df)
        df = map_columns(df, report_type)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df = df[df['date'] > '2000-01-01']
            df = df.sort_values('date')
            
        return df, None
    except Exception as e:
        return None, str(e)

def calculate_hedging_ranges(df):
    """
    ICT Logic: 
    1. Identify 6-Month (26 week) and 12-Month (52 week) High/Low of Net Hedger Position.
    2. Determine if we are in a Buy Program (>0) or Sell Program (<0).
    3. Identify 'Nodules': Turning points within the range.
    """
    df['Net Hedger'] = df['hedge_long'] - df['hedge_short']
    df['Net Speculator'] = df['spec_long'] - df['spec_short']

    # 6 Month Range (approx 26 weeks)
    df['6M_High'] = df['Net Hedger'].rolling(window=26).max()
    df['6M_Low'] = df['Net Hedger'].rolling(window=26).min()
    
    # 12 Month Range (approx 52 weeks)
    df['12M_High'] = df['Net Hedger'].rolling(window=52).max()
    df['12M_Low'] = df['Net Hedger'].rolling(window=52).min()
    
    # Range Oscillator (0 to 100%)
    df['Hedge_Index_6M'] = ((df['Net Hedger'] - df['6M_Low']) / (df['6M_High'] - df['6M_Low'])) * 100
    
    return df

def generate_ict_signal(row):
    """
    Synthesizes the 'Zero Line' (Macro) with the 'Range Index' (Micro).
    """
    net_pos = row['Net Hedger']
    hedge_idx = row['Hedge_Index_6M']
    
    # 1. Macro Program (Zero Line)
    macro_program = "BUY PROGRAM (Above Zero)" if net_pos > 0 else "SELL PROGRAM (Below Zero)"
    macro_color = "green" if net_pos > 0 else "red"
    
    # 2. Hedging Activity (Range Location)
    # ICT: Look for buying at the low of the range, selling at the high.
    if hedge_idx <= 15:
        hedging_action = "📈 Aggressive Accumulation (Bullish Nodule)"
        context = "Commercials are buying heavily relative to recent history."
    elif hedge_idx >= 85:
        hedging_action = "📉 Aggressive Distribution (Bearish Nodule)"
        context = "Commercials are selling heavily relative to recent history."
    else:
        hedging_action = "Passive / Neutral"
        context = "Commercials are maintaining positions within the range."
        
    return macro_program, macro_color, hedging_action, context

# --- ML HELPER FUNCTIONS ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_osok_ml_data(df):
    logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    w_df = df.groupby('Week_Start').agg(logic).sort_index()
    
    w_df['20W_High'] = w_df['High'].rolling(window=20).max()
    w_df['20W_Low'] = w_df['Low'].rolling(window=20).min()
    w_df['Equilibrium'] = (w_df['20W_High'] + w_df['20W_Low']) / 2
    w_df['PD_Factor'] = (w_df['Close'] - w_df['20W_Low']) / (w_df['20W_High'] - w_df['20W_Low'])
    w_df['Dist_Eq'] = (w_df['Close'] - w_df['Equilibrium']) / w_df['Equilibrium']
    w_df['RSI'] = calculate_rsi(w_df['Close'], 14)
    w_df['Prev_Ret'] = w_df['Close'].pct_change()
    
    # Trend Filters
    w_df['SMA_20'] = w_df['Close'].rolling(window=20).mean()
    w_df['SMA_50'] = w_df['Close'].rolling(window=50).mean()
    w_df['Trend_Bullish'] = (w_df['SMA_20'] > w_df['SMA_50']).astype(int)
    w_df['Month'] = w_df.index.month
    w_df['TR'] = np.maximum(w_df['High'] - w_df['Low'], np.abs(w_df['High'] - w_df['Close'].shift(1)))
    w_df['ATR'] = w_df['TR'].rolling(window=14).mean()
    w_df['Volatility'] = w_df['ATR'] / w_df['Close']
    
    # Target
    w_df['Target'] = (w_df['Close'].shift(-1) > w_df['Open'].shift(-1)).astype(int)
    
    return w_df.dropna()

def train_osok_model(ml_df):
    feature_cols = ['PD_Factor', 'Dist_Eq', 'RSI', 'Prev_Ret', 'Trend_Bullish', 'Month', 'Volatility']
    X = ml_df[feature_cols]
    y = ml_df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return model, acc, feature_cols

# --- WEEKLY ANALYSIS FUNCTIONS ---
def identify_weekly_profile(week_df):
    if week_df.empty: return {"Type": "Insufficient Data"}
    try:
        open_p = week_df.iloc[0]['Open'].item()
        close_p = week_df.iloc[-1]['Close'].item()
    except AttributeError:
        open_p = week_df.iloc[0]['Open']
        close_p = week_df.iloc[-1]['Close']
        
    is_bullish = close_p > open_p
    trend = "Bullish" if is_bullish else "Bearish"
    
    high_date = week_df.loc[week_df['High'].idxmax(), 'Date']
    low_date = week_df.loc[week_df['Low'].idxmin(), 'Date']
    if hasattr(high_date, 'item'): high_date = high_date.item()
    if hasattr(low_date, 'item'): low_date = low_date.item()
    
    high_day = high_date.weekday()
    low_day = low_date.weekday()
    
    profile, desc = "Undefined", "N/A"
    
    if is_bullish:
        if low_day == 1: profile, desc = "Classic Tuesday Low", "Low on Tue, expansion higher."
        elif low_day == 0: profile, desc = "Monday Low", "Low on Mon, steady expansion."
        elif low_day == 2: profile, desc = "Wednesday Low / Reversal", "Manipulation Mon-Tue, Low on Wed."
        elif low_day == 3: profile, desc = "Consolidation Thu Reversal", "Stop hunt Low on Thu, then Reversal."
        elif low_day == 4: profile, desc = "Seek & Destroy / Fri Low", "Choppy, Low late Friday."
    else:
        if high_day == 1: profile, desc = "Classic Tuesday High", "High on Tue, expansion lower."
        elif high_day == 0: profile, desc = "Monday High", "High on Mon, steady decline."
        elif high_day == 2: profile, desc = "Wednesday High / Reversal", "Manipulation Mon-Tue, High on Wed."
        elif high_day == 3: profile, desc = "Consolidation Thu Reversal", "Stop hunt High on Thu, then Reversal."
        elif high_day == 4: profile, desc = "Seek & Destroy / Fri High", "Choppy, High late Friday."
        
    return {"Trend": trend, "Profile": profile, "Description": desc, "Weekly_High": week_df['High'].max(), "Weekly_Low": week_df['Low'].min(), "High_Day": high_date.strftime("%A"), "Low_Day": low_date.strftime("%A"), "High_Day_Num": high_day, "Low_Day_Num": low_day}

def calculate_seasonal_path(weeks_dict, lookback):
    seasonal_data = []
    week_keys = sorted(list(weeks_dict.groups.keys()), reverse=True)
    for w_start in week_keys[:lookback]:
        w_df = weeks_dict.get_group(w_start).copy()
        if len(w_df) < 2: continue
        mon_open = w_df.iloc[0]['Open']
        if hasattr(mon_open, 'item'): mon_open = mon_open.item()
        for _, row in w_df.iterrows():
            seasonal_data.append({"DayNum": row['Date'].weekday(), "PctChange": ((row['Close'] - mon_open) / mon_open) * 100, "DayName": row['Date'].strftime("%A")})
    if not seasonal_data: return None
    return pd.DataFrame(seasonal_data).groupby('DayNum').agg({'PctChange': 'mean', 'DayName': 'first'}).reset_index()

def predict_next_week(stats_df, current_profile):
    if stats_df.empty: return None
    df_sorted = stats_df.sort_values('Week Start', ascending=True).copy()
    df_sorted['Next_Profile'] = df_sorted['Profile'].shift(-1)
    transitions = df_sorted[df_sorted['Profile'] == current_profile]
    if transitions.empty or transitions['Next_Profile'].dropna().empty: return None
    counts = transitions['Next_Profile'].value_counts(normalize=True)
    return dict(sorted(counts.to_dict().items(), key=lambda item: item[1], reverse=True))

def identify_intraday_profile(df):
    if df.empty: return None
    midnight_bar = df[df['Datetime_NY'].dt.hour == 0]
    if midnight_bar.empty: midnight_open = df.iloc[0]['Open']
    else: midnight_open = midnight_bar.iloc[0]['Open']
        
    judas_start = time(0, 0)
    judas_end = time(2, 0)
    current_price = df.iloc[-1]['Close']
    is_bullish = current_price > midnight_open
    trend = "Bullish" if is_bullish else "Bearish"
    
    day_high = df['High'].max()
    day_low = df['Low'].min()
    high_time = df.loc[df['High'].idxmax(), 'Datetime_NY'].time()
    low_time = df.loc[df['Low'].idxmin(), 'Datetime_NY'].time()
    
    day_range = day_high - day_low
    threshold = day_range * 0.10
    
    profile, desc = "Consolidation", "Choppy."
    
    if is_bullish:
        if judas_start <= low_time <= judas_end: 
            if (current_price - day_low) > threshold:
                profile, desc = "London Normal (Buy)", "Judas Swing Low (0-2 AM) with Displacement."
            else:
                profile, desc = "Weak London", "Low formed in KZ, but no expansion yet."
        elif low_time > judas_end: profile, desc = "London Delayed (Buy)", "Low formed after 2 AM."
    else:
        if judas_start <= high_time <= judas_end: 
            if (day_high - current_price) > threshold:
                profile, desc = "London Normal (Sell)", "Judas Swing High (0-2 AM) with Displacement."
            else:
                profile, desc = "Weak London", "High formed in KZ, but no expansion yet."
        elif high_time > judas_end: profile, desc = "London Delayed (Sell)", "High formed after 2 AM."

    return {"Trend": trend, "Profile": profile, "Description": desc, "Midnight_Open": midnight_open, "High": day_high, "Low": day_low, "High_Time": high_time, "Low_Time": low_time}

# =========================================
# MAIN LOGIC BRANCHING
# =========================================

if analysis_mode == "Weekly Protocols (News & Logic)":
    
    st.sidebar.subheader("Protocol Settings")
    sim_mode = st.sidebar.checkbox("Simulate Jan 3-9, 2026 Context", value=True, help="Forces the logic to run on the specific date mentioned.")
    
    st.title("📅 Weekly Profile Protocols")
    st.markdown("Automated **Decision Tree** based on Economic Calendar & News Sentiment.")
    
    with st.spinner("Scraping Forex Factory & News Feeds..."):
        if sim_mode:
            stories = [
                {"Title": "Oil prices steady amidst quiet trading", "Publisher": "Reuters", "Time": "2026-01-03 10:00"},
                {"Title": "Post-holiday liquidity remains thin", "Publisher": "Bloomberg", "Time": "2026-01-03 09:30"}
            ]
            ff_df = pd.DataFrame(columns=["Date", "Time", "Currency", "Event", "Impact"]) 
            current_date_ref = datetime(2026, 1, 5).date() 
        else:
            ff_df = scrape_forex_factory(week="this")
            stories = get_financial_news()
            current_date_ref = datetime.now().date()
            
    protocol, clustering, days_with_news, exo_found = determine_weekly_protocol(ff_df, stories, manual_override_date=current_date_ref)
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Current Day", current_date_ref.strftime("%A, %b %d"))
    col_stat2.metric("News Clustering", clustering, delta="High Impact" if clustering != "Minimal/None" else "Low Vol", delta_color="off")
    col_stat3.metric("Exogenous Shock?", "YES" if exo_found else "NO", delta="Override Active" if exo_found else "Normal", delta_color="inverse" if exo_found else "off")
    
    st.divider()
    
    action_color = THEME['bearish'] if "Avoid" in protocol['Action'] else THEME['bullish']
    st.markdown(f"""
    <div style="background-color: #1E252F; border-left: 5px solid {action_color}; padding: 20px; border-radius: 5px;">
        <h2 style="margin:0; color: white;">{protocol['Action']}</h2>
        <h4 style="margin:5px 0; color: #CCCCCC;">Profile: {protocol['Profile']}</h4>
        <p style="margin-top: 10px; font-style: italic;">"{protocol['Logic']}"</p>
        <hr style="border-color: #333;">
        <strong>Strategic Bias:</strong> {protocol['Bias']}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧩 Decision Tree Logic")
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.subheader("📰 Economic Calendar (High Impact)")
        if ff_df is not None and not ff_df.empty:
            display_df = ff_df[['Date', 'Time', 'Currency', 'Event']].copy()
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No High Impact (Red Folder) USD events scheduled.")
    with col_d2:
        st.subheader("📢 Hottest Stories (Sentiment)")
        if stories:
            for s in stories:
                st.markdown(f"**[{s['Title']}]({s.get('Link', '#')})**")
                st.caption(f"{s['Publisher']} • {s['Time']}")
        else:
            st.warning("No news headlines found.")

elif analysis_mode == "Weekly Profiles":
    
    st.sidebar.subheader("Weekly Settings")
    lookback_weeks = st.sidebar.slider("Weeks to Display", 1, 20, 4)
    stats_lookback = st.sidebar.slider("Stats Range (Prediction History)", 52, 300, 150)
    
    st.title(f"📊 Weekly Profile Identifier: {selected_asset_name}")
    
    with st.spinner(f"Fetching weekly data for {ticker_symbol}..."):
        df = get_data_weekly(ticker_symbol, max(lookback_weeks, stats_lookback) + 3)

    if df is not None:
        weeks = df.groupby('Week_Start')
        week_keys = sorted(list(weeks.groups.keys()), reverse=True)
        week_keys = [k for k in week_keys if not pd.isna(k)]
        
        if not week_keys:
            st.error("No data.")
        else:
            stats_data = []
            for w_start in week_keys[:stats_lookback]:
                w_df = weeks.get_group(w_start)
                if len(w_df) >= 3:
                    res = identify_weekly_profile(w_df)
                    res['Week Start'] = w_start
                    stats_data.append(res)
            stats_df = pd.DataFrame(stats_data)

            tab1, tab2 = st.tabs(["🔎 Current Analysis & Prediction", "📈 Statistical Probability"])
            
            with tab1:
                col_sel, col_blank = st.columns([1, 2])
                with col_sel:
                    sel_week = st.selectbox("Select Week", week_keys[:lookback_weeks], format_func=lambda x: f"Week of {x.strftime('%Y-%m-%d')}")
                
                sel_idx = week_keys.index(sel_week)
                prev_data = None
                if sel_idx + 1 < len(week_keys):
                    p_df = weeks.get_group(week_keys[sel_idx+1])
                    prev_data = {"PWH": p_df['High'].max(), "PWL": p_df['Low'].min()}
                
                curr_df = weeks.get_group(sel_week).copy()
                analysis = identify_weekly_profile(curr_df)
                
                st.markdown("### 🔮 Predictive Analysis")
                container = st.container()
                with container:
                    if not stats_df.empty:
                        prediction = predict_next_week(stats_df, analysis['Profile'])
                        col_pred1, col_pred2 = st.columns([1, 2])
                        with col_pred1:
                            st.info(f"**Current Profile:**\n\n{analysis['Profile']}")
                        with col_pred2:
                            if prediction:
                                st.write(f"**Historical Probability (last {len(stats_df)} weeks):**")
                                for next_prof, prob in list(prediction.items())[:3]:
                                    st.caption(f"{next_prof} ({prob*100:.1f}%)")
                                    st.progress(prob)
                            else:
                                st.warning("Not enough historical data to predict next week.")
                
                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Trend", analysis["Trend"], delta_color="normal" if analysis["Trend"]=="Bullish" else "inverse")
                c2.metric("Profile", analysis["Profile"])
                c3.metric("High Day", analysis["High_Day"])
                c4.metric("Low Day", analysis["Low_Day"])
                st.caption(f"**Logic:** {analysis['Description']}")
                
                fig = go.Figure(data=[go.Candlestick(
                    x=curr_df['Date'], open=curr_df['Open'], high=curr_df['High'], 
                    low=curr_df['Low'], close=curr_df['Close'],
                    increasing_line_color=THEME['bullish'], decreasing_line_color=THEME['bearish']
                )])
                
                if prev_data:
                    fig.add_hline(y=prev_data['PWH'], line_dash="dash", line_color=THEME['text'], opacity=0.5, annotation_text="PWH")
                    fig.add_hline(y=prev_data['PWL'], line_dash="dash", line_color=THEME['text'], opacity=0.5, annotation_text="PWL", annotation_position="bottom right")
                
                fig.update_layout(title=f"Weekly Chart: {selected_asset_name}", height=600, xaxis_rangeslider_visible=False)
                fig = apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                if not stats_df.empty:
                    st.subheader("History")
                    st.dataframe(stats_df[['Week Start', 'Trend', 'Profile', 'High_Day', 'Low_Day']].head(lookback_weeks), use_container_width=True)

            with tab2:
                if not stats_df.empty:
                    st.subheader("Average Weekly Path")
                    avg_path = calculate_seasonal_path(weeks, stats_lookback)
                    if avg_path is not None:
                        fig_seas = px.line(avg_path, x="DayName", y="PctChange", markers=True, title="Composite Weekly Path (% vs Mon Open)")
                        fig_seas.update_traces(line_color=THEME['line_primary'])
                        fig_seas.add_hline(y=0, line_dash="dot", line_color=THEME['text'])
                        fig_seas = apply_theme(fig_seas)
                        st.plotly_chart(fig_seas, use_container_width=True)
                    
                    st.divider()
                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.subheader("Day Probability")
                        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                        h_c = stats_df['High_Day'].value_counts().reindex(days, fill_value=0)
                        l_c = stats_df['Low_Day'].value_counts().reindex(days, fill_value=0)
                        df_c = pd.DataFrame({"Day": days, "Highs": h_c.values, "Lows": l_c.values}).melt(id_vars="Day", var_name="Type", value_name="Count")
                        fig_d = px.bar(df_c, x="Day", y="Count", color="Type", barmode="group", color_discrete_map={"Highs": THEME['bullish'], "Lows": THEME['bearish']})
                        fig_d = apply_theme(fig_d)
                        st.plotly_chart(fig_d, use_container_width=True)
                    with c_b:
                        st.subheader("Profile Distribution")
                        profile_counts = stats_df['Profile'].value_counts().reset_index()
                        profile_counts.columns = ['Profile', 'Count'] 
                        fig_p = px.pie(profile_counts, names='Profile', values='Count', hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
                        fig_p = apply_theme(fig_p)
                        st.plotly_chart(fig_p, use_container_width=True)

elif analysis_mode == "Intraday Profiles":
    
    st.sidebar.subheader("Intraday Settings")
    today = datetime.now().date()
    min_date = today - timedelta(days=59)
    target_date = st.sidebar.date_input("Select Trading Day", today, min_value=min_date, max_value=today)
    
    show_killzones = st.sidebar.checkbox("Show Kill Zones (London/NY)", value=True)
    show_pdl_pdh = st.sidebar.checkbox("Show Previous Day High/Low", value=True)
    
    st.title(f"⏱️ Intraday Profile (London): {selected_asset_name}")
    st.markdown(f"Analyzing London Protraction logic for **{target_date}**.")
    
    with st.spinner("Fetching 5-minute data..."):
        df_intra = get_data_intraday(ticker_symbol, target_date)
        
    if df_intra is not None and not df_intra.empty:
        res = identify_intraday_profile(df_intra)
        
        prev_day_stats = None
        if show_pdl_pdh:
            daily_check = yf.download(ticker_symbol, period="5d", interval="1d", progress=False)
            if not daily_check.empty:
                if isinstance(daily_check.columns, pd.MultiIndex): daily_check.columns = daily_check.columns.get_level_values(0)
                daily_check = daily_check.reset_index()
                daily_check['Date'] = pd.to_datetime(daily_check['Date']).dt.date
                past_days = daily_check[daily_check['Date'] < target_date]
                if not past_days.empty:
                    last_day = past_days.iloc[-1]
                    try: pdh, pdl = last_day['High'].item(), last_day['Low'].item()
                    except: pdh, pdl = last_day['High'], last_day['Low']
                    prev_day_stats = {"PDH": pdh, "PDL": pdl}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trend", res['Trend'], delta_color="normal" if res['Trend']=="Bullish" else "inverse")
        c2.metric("Profile", res['Profile'])
        c3.metric("High Time", str(res['High_Time'])[:5])
        c4.metric("Low Time", str(res['Low_Time'])[:5])
        st.info(f"**Scenario:** {res['Description']}")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_intra['Datetime_NY'], open=df_intra['Open'], high=df_intra['High'], 
            low=df_intra['Low'], close=df_intra['Close'], name="Price (5m)",
            increasing_line_color=THEME['bullish'], decreasing_line_color=THEME['bearish']
        ))
        fig.add_hline(y=res['Midnight_Open'], line_dash="dot", line_color="white", annotation_text="Midnight Open")
        
        base_dt = pd.to_datetime(target_date).tz_localize('America/New_York') if df_intra['Datetime_NY'].iloc[0].tzinfo else pd.to_datetime(target_date)
        def to_ms(h, m): return base_dt.replace(hour=h, minute=m).timestamp() * 1000

        fig.add_vline(x=to_ms(0,0), line_dash="solid", line_color=THEME['grid'], annotation_text="00:00 NY")
        fig.add_vline(x=to_ms(2,0), line_dash="solid", line_color=THEME['grid'], annotation_text="02:00 NY")
        
        if show_killzones:
            fig.add_vrect(x0=to_ms(2,0), x1=to_ms(5,0), fillcolor=THEME['bullish'], opacity=0.1, annotation_text="London KZ", annotation_position="top left")
            fig.add_vrect(x0=to_ms(7,0), x1=to_ms(10,0), fillcolor=THEME['bearish'], opacity=0.1, annotation_text="NY KZ", annotation_position="top left")

        if prev_day_stats:
            fig.add_hline(y=prev_day_stats['PDH'], line_dash="dash", line_color=THEME['bullish'], opacity=0.6, annotation_text="PDH")
            fig.add_hline(y=prev_day_stats['PDL'], line_dash="dash", line_color=THEME['bearish'], opacity=0.6, annotation_text="PDL", annotation_position="bottom right")

        fig.update_layout(title=f"Intraday Chart (NY Time) - {target_date}", height=600, xaxis_rangeslider_visible=False)
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        
    else: st.error("No Intraday data found.")

elif analysis_mode == "One Shot One Kill (OSOK)":
    
    st.sidebar.subheader("OSOK Settings")
    ote_anchor = st.sidebar.radio("OTE Fib Anchor", ["Current Day Range", "Previous Week Range"], help="Determines the High/Low used for the OTE Fib calculation.")
    
    st.title("🎯 One Shot One Kill (OSOK) + ML Model")
    st.markdown("Identifies the **20-Week Dealing Range** and uses **Machine Learning** to predict the bias.")

    with st.spinner("Analyzing 20-Week IPDA Range..."):
        df_full = get_data_weekly(ticker_symbol, weeks=300)
    
    if df_full is not None:
        df_weekly = df_full.sort_values('Date')
        last_20 = df_weekly.iloc[-21:-1]
        current_week = df_weekly.iloc[-1]
        
        ipda_high = last_20['High'].max()
        ipda_low = last_20['Low'].min()
        ipda_range = ipda_high - ipda_low
        equilibrium = (ipda_high + ipda_low) / 2
        
        current_close = current_week['Close']
        in_premium = current_close > equilibrium
        
        m1, m2, m3 = st.columns(3)
        m1.metric("20-Week High (Liquidity)", f"{ipda_high:.2f}")
        m2.metric("20-Week Low (Liquidity)", f"{ipda_low:.2f}")
        m3.metric("Current State", "PREMIUM (Sell)" if in_premium else "DISCOUNT (Buy)", 
                  delta="-Sell" if in_premium else "+Buy", delta_color="inverse")
        
        st.markdown("### Range Position")
        st.progress((current_close - ipda_low) / ipda_range)
        st.caption(f"Price is at {((current_close - ipda_low) / ipda_range)*100:.1f}% of the 20-week range.")

        st.divider()
        st.subheader("🤖 OSOK ML Probability Engine")
        
        with st.expander("View ML Model Logic"):
            st.markdown("""
            **How this ML Model works:**
            1. **Feature Extraction:** It calculates where price is within the 20-week range (Premium/Discount factor).
            2. **Momentum:** RSI and recent weekly returns.
            3. **Trend/Seasonality:** Checks 20 vs 50 EMA and current Month.
            4. **Target:** Trains a Random Forest to predict if the *Next Week* closes higher than it opens.
            """)
        
        ml_df = prepare_osok_ml_data(df_full)
        
        if len(ml_df) > 50:
            model, acc, feats = train_osok_model(ml_df)
            latest_features = ml_df.iloc[[-1]][feats]
            pred_prob = model.predict_proba(latest_features)[0] 
            
            ml_c1, ml_c2 = st.columns(2)
            with ml_c1:
                st.metric("Model Backtest Accuracy", f"{acc*100:.1f}%")
                bias_score = pred_prob[1]
                if bias_score > 0.55: st.success(f"**ML Bias: BULLISH ({bias_score*100:.1f}%)**")
                elif bias_score < 0.45: st.error(f"**ML Bias: BEARISH ({(1-bias_score)*100:.1f}%)**")
                else: st.warning(f"**ML Bias: NEUTRAL / CHOPPY**")
            with ml_c2:
                importances = pd.DataFrame({'Feature': feats, 'Importance': model.feature_importances_})
                fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', title="Factor Importance")
                fig_imp.update_traces(marker_color=THEME['line_primary'])
                fig_imp = apply_theme(fig_imp)
                fig_imp.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning("Not enough historical data to train the OSOK ML model reliably.")

        st.divider()
        st.subheader("Execution: 15m OTE Setup")
        
        today = datetime.now().date()
        target_date_osok = st.sidebar.date_input("Select Day", today)
        df_15m = get_data_intraday(ticker_symbol, target_date_osok, interval="15m")
        
        if df_15m is not None and not df_15m.empty:
            day_name = pd.to_datetime(target_date_osok).strftime("%A")
            is_anchor_day = day_name in ["Monday", "Tuesday", "Wednesday"]
            
            status_color = "green" if is_anchor_day else "orange"
            st.markdown(f"**Day:** {day_name} (:{status_color}[{'Anchor Point Potential' if is_anchor_day else 'Standard Trading Day'}])")
            
            fig_osok = go.Figure()
            fig_osok.add_trace(go.Candlestick(
                x=df_15m['Datetime_NY'], open=df_15m['Open'], high=df_15m['High'], 
                low=df_15m['Low'], close=df_15m['Close'], name="Price (15m)",
                increasing_line_color=THEME['bullish'], decreasing_line_color=THEME['bearish']
            ))
            
            if ote_anchor == "Previous Week Range":
                view_high = ipda_high
                view_low = ipda_low
            else:
                view_high = df_15m['High'].max()
                view_low = df_15m['Low'].min()
                
            diff = view_high - view_low
            
            if in_premium:
                ote_62 = view_low + (diff * 0.62)
                ote_79 = view_low + (diff * 0.79)
                color_ote = THEME['bearish']
                bias_text = "Bearish OTE Zone (Sell)"
            else:
                ote_62 = view_high - (diff * 0.62)
                ote_79 = view_high - (diff * 0.79)
                color_ote = THEME['bullish']
                bias_text = "Bullish OTE Zone (Buy)"
            
            fig_osok.add_hrect(y0=ote_62, y1=ote_79, fillcolor=color_ote, opacity=0.1, annotation_text=bias_text, annotation_position="right")
            
            base_dt_osok = pd.to_datetime(target_date_osok).tz_localize('America/New_York') if df_15m['Datetime_NY'].iloc[0].tzinfo else pd.to_datetime(target_date_osok)
            def to_ms_osok(h, m): return base_dt_osok.replace(hour=h, minute=m).timestamp() * 1000
            
            fig_osok.add_vrect(x0=to_ms_osok(2,0), x1=to_ms_osok(5,0), fillcolor=THEME['bullish'], opacity=0.07, annotation_text="London KZ", annotation_position="top left")
            fig_osok.add_vrect(x0=to_ms_osok(7,0), x1=to_ms_osok(10,0), fillcolor=THEME['bearish'], opacity=0.07, annotation_text="NY KZ", annotation_position="top left")

            fig_osok.update_layout(title=f"OSOK Execution (15m) - {target_date_osok}", height=600, xaxis_rangeslider_visible=False)
            fig_osok = apply_theme(fig_osok)
            st.plotly_chart(fig_osok, use_container_width=True)
            
            st.info(f"**OSOK Checklist:** 1. Price in { 'Premium' if in_premium else 'Discount' } of 20-week range? ✅ | 2. Is it Mon/Tue/Wed? {'✅' if is_anchor_day else '❌'} | 3. Wait for price to hit the OTE Zone during a Kill Zone.")
        else:
            st.warning("No intraday data available for the selected date.")
    else:
        st.error("Could not fetch weekly data.")

elif analysis_mode == "COT Quant Terminal":
    
    st.title("⚡ COT Quant Terminal")
    st.markdown("Automated **Commercial Hedging Program** detection using CFTC Data.")
    
    # Configuration is set in sidebar, accessing it here
    config = COT_ASSET_CONFIG[selected_cot_asset]
    spec_label, hedge_label = config['labels']

    @st.cache_data(ttl=3600)
    def fetch_multi_year(start_y, end_y, r_type):
        master_df = pd.DataFrame()
        for y in range(start_y, end_y + 1):
            df, err = load_cot_data(y, r_type)
            if df is not None:
                master_df = pd.concat([master_df, df])
        return master_df

    with st.spinner(f"Loading COT data for {selected_cot_asset}..."):
        df_raw = fetch_multi_year(cot_start_year, current_year, config['report_type'])

    # Filter Asset
    if not df_raw.empty:
        keywords = config['keywords']
        mask = df_raw['market'].astype(str).apply(lambda x: all(k.lower() in x.lower() for k in keywords))
        df_asset = df_raw[mask].copy().sort_values('date')

        if df_asset.empty:
            st.warning(f"No data found for {selected_cot_asset}. Check keywords.")
        else:
            # Calculations
            if all(c in df_asset.columns for c in ['spec_long', 'spec_short', 'hedge_long', 'hedge_short']):
                df_asset = calculate_hedging_ranges(df_asset)
                
                latest = df_asset.iloc[-1]
                prev = df_asset.iloc[-2]
                
                macro_prog, macro_col, hedge_act, hedge_ctx = generate_ict_signal(latest)
                
                # --- DASHBOARD ---
                st.markdown(f"### 🧬 ICT Hedging Program Analysis ({hedge_label})")
                
                # Top Level Status
                status_col1, status_col2 = st.columns([1, 2])
                
                with status_col1:
                    st.markdown(f"**Macro Bias (Zero Line)**")
                    st.markdown(f":{macro_col}[**{macro_prog}**]")
                    st.caption(f"Net Position: {int(latest['Net Hedger']):,}")
                    
                with status_col2:
                    st.markdown(f"**Hedging Activity (6-Month Range)**")
                    st.markdown(f"**{hedge_act}**")
                    st.caption(hedge_ctx)
                    
                st.divider()

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Commercial Net", f"{int(latest['Net Hedger']):,}", f"{int(latest['Net Hedger'] - prev['Net Hedger']):,}")
                m2.metric("6-Month Range High", f"{int(latest['6M_High']):,}", help="Highest Net Position in last 26 weeks")
                m3.metric("6-Month Range Low", f"{int(latest['6M_Low']):,}", help="Lowest Net Position in last 26 weeks")
                m4.metric("Range Position", f"{latest['Hedge_Index_6M']:.1f}%", help="0% = At Lows (Buying), 100% = At Highs (Selling)")

                # --- VISUALIZATIONS ---
                tab1, tab2, tab3 = st.tabs(["📊 Commercial Hedging Program", "🦋 Net Structure", "⚡ Speculator Sentiment"])
                
                with tab1:
                    st.markdown("**Instructions:** Focus on the Red Line (Commercials). The Shaded Area represents the 12-Month Range.")
                    
                    fig1 = go.Figure()
                    
                    # 1. The Range (Background) - Represents the "12 Month Range" concept
                    fig1.add_trace(go.Scatter(
                        x=df_asset['date'], y=df_asset['12M_High'],
                        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                    ))
                    fig1.add_trace(go.Scatter(
                        x=df_asset['date'], y=df_asset['12M_Low'],
                        mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)',
                        name='12M Range', hoverinfo='skip'
                    ))
                    
                    # 2. The Commercial Line (The "Red Line") 
                    fig1.add_trace(go.Scatter(
                        x=df_asset['date'], y=df_asset['Net Hedger'],
                        name=f'Commercials ({hedge_label})',
                        line=dict(color='#FF0000', width=3)
                    ))
                    
                    # 3. The Zero Line
                    fig1.add_hline(y=0, line_dash="solid", line_color="white", annotation_text="Zero Line Basis", annotation_position="bottom right")
                    
                    fig1.update_layout(title="Commercial Hedging Program (Net Trader Position)", height=600, hovermode="x unified")
                    fig1 = apply_theme(fig1)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                with tab2:
                    # Butterfly Chart to show total Open Interest structure
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(x=df_asset['date'], y=df_asset['hedge_long'], name=f"{hedge_label} Longs", marker_color='rgba(255, 0, 0, 0.6)'))
                    fig2.add_trace(go.Bar(x=df_asset['date'], y=-df_asset['hedge_short'], name=f"{hedge_label} Shorts", marker_color='rgba(255, 0, 0, 0.3)'))
                    fig2.add_trace(go.Scatter(x=df_asset['date'], y=df_asset['Net Hedger'], name="Net Position", line=dict(color='white', width=2, dash='dot')))
                    
                    fig2.update_layout(title="Commercial Structure (Longs vs Shorts)", barmode='overlay', height=500, hovermode="x unified")
                    fig2 = apply_theme(fig2)
                    st.plotly_chart(fig2, use_container_width=True)

                with tab3:
                    # Speculator view for "Dumb Money" contrast 
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(x=df_asset['date'], y=df_asset['Net Speculator'], name='Large Speculators', line=dict(color='green', width=2)))
                    fig3.add_hline(y=0, line_dash="dot", line_color="gray")
                    fig3.update_layout(title="Large Speculator Net Position", height=500)
                    fig3 = apply_theme(fig3)
                    st.plotly_chart(fig3, use_container_width=True)

                with st.expander("View Raw Data"):
                    st.dataframe(df_asset.sort_values('date', ascending=False))
            else:
                st.error("Required columns missing for visualization.")
    else:
        st.error("Could not fetch COT data.")

# --- MARKET SCANNER ---
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Market Scanner")
if st.sidebar.button("Scan All Assets"):
    st.markdown("### 🔍 Market Wide Scan")
    results = []
    bar = st.progress(0)
    for i, (name, tk) in enumerate(asset_map.items()):
        bar.progress((i+1)/len(asset_map))
        try:
            if analysis_mode == "One Shot One Kill (OSOK)":
                 d = get_data_weekly(tk, 25)
                 if d is not None:
                      last_20 = d.iloc[-21:-1]
                      curr = d.iloc[-1]['Close']
                      high, low = last_20['High'].max(), last_20['Low'].min()
                      state = "Premium (Sell)" if curr > (high+low)/2 else "Discount (Buy)"
                      results.append({"Asset": name, "Type": "OSOK", "State": state, "20-Wk High": high, "20-Wk Low": low})
            elif analysis_mode == "Weekly Profiles":
                d = get_data_weekly(tk, 4)
                if d is not None:
                    last_wk = d.groupby('Week_Start').get_group(sorted(list(d['Week_Start'].unique()))[-1])
                    r = identify_weekly_profile(last_wk)
                    results.append({"Asset": name, "Type": "Weekly", "Profile": r['Profile'], "Trend": r['Trend']})
            elif analysis_mode == "COT Quant Terminal":
                # Skip scan for COT to avoid API rate limits/time expense during quick scan
                pass 
            else:
                d = get_data_intraday(tk, datetime.now().date())
                if d is not None:
                    r = identify_intraday_profile(d)
                    results.append({"Asset": name, "Type": "Intraday", "Profile": r['Profile'], "Trend": r['Trend']})
        except: pass
    bar.empty()
    if results:
        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True)
    elif analysis_mode == "COT Quant Terminal":
        st.info("Scanner not available for COT mode to preserve performance.")
    else:
        st.warning("No data found.")
