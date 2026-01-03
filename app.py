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
import cot_reports as cot  # NEW IMPORT FOR TRANSCRIPT LOGIC

# ML Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
        "COT Quant Terminal"  # NEW MODE BASED ON TRANSCRIPT
    ], 
    help="Select the analytical framework."
)

st.sidebar.markdown("---")

# 2. Asset Selection Logic (Conditional based on Mode)
if analysis_mode == "COT Quant Terminal":
    # COT CONFIGURATION (Transcript Specific)
    COT_ASSET_CONFIG = {
        "EUR/USD (6E)": {
            "keywords": ["EURO FX", "CHICAGO MERCANTILE EXCHANGE"],
            "report_type": "traders_in_financial_futures_fut",
            "labels": ("Leveraged Funds", "Asset Manager/Institutional")
        },
        "GBP/USD (6B)": {
            "keywords": ["BRITISH POUND", "CHICAGO MERCANTILE EXCHANGE"],
            "report_type": "traders_in_financial_futures_fut",
            "labels": ("Leveraged Funds", "Asset Manager/Institutional")
        },
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
        "Japanese Yen (6J)": {
            "keywords": ["JAPANESE YEN", "CHICAGO MERCANTILE EXCHANGE"],
            "report_type": "traders_in_financial_futures_fut",
            "labels": ("Leveraged Funds", "Asset Manager/Institutional")
        }
    }
    
    selected_cot_asset = st.sidebar.selectbox("Select Asset (COT)", list(COT_ASSET_CONFIG.keys()))
    current_year = datetime.now().year
    cot_start_year = st.sidebar.number_input("Lookback Start Year", min_value=2015, max_value=current_year, value=2024)
    ticker_symbol = "EURUSD=X" # Placeholder for charts
    
else:
    # STANDARD CONFIGURATION
    use_etf = st.sidebar.checkbox("Use ETF Tickers (More Stable)", value=False, 
        help="Check this if Futures data (ES=F, GC=F) fails to load.")

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
    """Scrapes Forex Factory Calendar for High Impact (Red) News."""
    url = f"https://www.forexfactory.com/calendar?week={week}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return pd.DataFrame()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find("table", class_="calendar__table")
        if not table: return pd.DataFrame()
        
        data = []
        rows = table.find_all("tr", class_=lambda x: x and "calendar__row" in x)
        current_date = None
        for row in rows:
            date_cell = row.find("td", class_="calendar__date")
            if date_cell and date_cell.text.strip(): current_date = date_cell.text.strip()
            currency = row.find("td", class_="calendar__currency").text.strip()
            impact_cell = row.find("td", class_="calendar__impact")
            impact_span = impact_cell.find("span") if impact_cell else None
            impact_class = impact_span.get("class", []) if impact_span else []
            impact = "High" if "icon--impact-red" in impact_class else "Low"
            event_name = row.find("td", class_="calendar__event").text.strip()
            time_str = row.find("td", class_="calendar__time").text.strip()
            if impact == "High":
                data.append({"Date": current_date, "Time": time_str, "Currency": currency, "Event": event_name, "Impact": impact})
        return pd.DataFrame(data)
    except: return pd.DataFrame()

def get_financial_news():
    try:
        ticker = yf.Ticker("EURUSD=X") 
        news = ticker.news
        stories = []
        for n in news[:5]:
            stories.append({"Title": n.get('title', "No Title"), "Publisher": n.get('publisher', "Unknown"), "Link": n.get('link', "#"), "Time": datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M')})
        return stories
    except: return []

def determine_weekly_protocol(ff_df, stories, manual_override_date=None):
    today = manual_override_date if manual_override_date else datetime.now().date()
    day_name = today.strftime("%A")
    if ff_df is not None and not ff_df.empty: usd_news = ff_df[ff_df['Currency'] == 'USD']
    else: usd_news = pd.DataFrame()
    clustering = "Minimal/None"
    red_folder_days = []
    if not usd_news.empty:
        usd_news['DayName'] = usd_news['Date'].apply(lambda x: x.split(' ')[0] if x else "")
        red_folder_days = usd_news['DayName'].unique().tolist()
        has_early = any(d in ['Mon', 'Tue'] for d in red_folder_days)
        has_mid = any(d in ['Wed', 'Thu'] for d in red_folder_days)
        if has_early and not has_mid: clustering = "Early Week (Mon-Tue)"
        elif has_mid: clustering = "Midweek (Tue-Thu)"
    
    exogenous_found = False
    exo_headline = ""
    keywords = ["War", "Sanctions", "Invasion", "Emergency", "Crisis"]
    if stories:
        for s in stories:
            if any(k.lower() in s.get('Title', "").lower() for k in keywords):
                exogenous_found = True; exo_headline = s.get('Title'); break

    protocol = {"Action": "Consolidation", "Profile": "Neutral", "Logic": "Standard", "Bias": "Neutral"}
    if day_name == "Monday":
        if exogenous_found: protocol = {"Action": "⚠️ ALERT: Volatility Injection", "Profile": "Seek & Destroy", "Logic": f"Catalyst: {exo_headline}", "Bias": "Follow Displacement"}
        else: protocol = {"Action": "⛔ WAIT: Avoid Monday", "Profile": "Classic Consolidation", "Logic": "No high-impact drivers.", "Bias": "Range Bound"}
    elif clustering == "Midweek (Tue-Thu)":
        protocol = {"Action": "Wait for Midweek Manipulation", "Profile": "Classic Expansion", "Logic": "Tue/Wed news acts as Judas Swing.", "Bias": "Reversal likely Wed"}
        
    return protocol, clustering, red_folder_days, exogenous_found

# --- SHARED HELPER FUNCTIONS (YFINANCE) ---

def get_data_weekly(ticker, weeks=52):
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
        data['Week_Start'] = data['Date'].apply(lambda x: x - timedelta(days=x.weekday()))
        return data
    except Exception as e: return None

def get_data_intraday(ticker, target_date, interval="5m"):
    start_date = target_date - timedelta(days=2) 
    end_date = target_date + timedelta(days=2)
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        if data['Datetime'].dt.tz is None: data['Datetime'] = data['Datetime'].dt.tz_localize('UTC')
        else: data['Datetime'] = data['Datetime'].dt.tz_convert('UTC')
        data['Datetime_NY'] = data['Datetime'].dt.tz_convert('America/New_York')
        mask = data['Datetime_NY'].dt.date == target_date
        return data.loc[mask].copy()
    except Exception as e: return None

# --- SHARED HELPER FUNCTIONS (COT DATA - TRANSCRIPT LOGIC) ---

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

    if "disaggregated" in report_type:
        col_map['spec_long'] = get_col(['money', 'long'], exclude=['lev'])
        col_map['spec_short'] = get_col(['money', 'short'], exclude=['lev'])
        col_map['hedge_long'] = get_col(['prod', 'merc', 'long'])
        col_map['hedge_short'] = get_col(['prod', 'merc', 'short'])
        
    elif "financial" in report_type:
        col_map['spec_long'] = get_col(['lev', 'money', 'long'])
        col_map['spec_short'] = get_col(['lev', 'money', 'short'])
        col_map['hedge_long'] = get_col(['dealer', 'long']) or get_col(['asset', 'long'])
        col_map['hedge_short'] = get_col(['dealer', 'short']) or get_col(['asset', 'short'])

    final_map = {v: k for k, v in col_map.items() if v}
    df = df.rename(columns=final_map)
    for c in ['spec_long', 'spec_short', 'hedge_long', 'hedge_short']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
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
            df = df.sort_values('date')
        return df, None
    except Exception as e: return None, str(e)

def calculate_hedging_ranges(df):
    """
    ICT Logic from Transcript: 
    1. Identify 12-Month (52 week) High/Low of Net Hedger Position.
    2. Determine Buy/Sell Program based on Zero Line.
    """
    df['Net Hedger'] = df['hedge_long'] - df['hedge_short']
    df['Net Speculator'] = df['spec_long'] - df['spec_short']

    # 12 Month Range (approx 52 weeks) - "Look at the last year"
    df['12M_High'] = df['Net Hedger'].rolling(window=52).max()
    df['12M_Low'] = df['Net Hedger'].rolling(window=52).min()
    
    # Range Oscillator (0 to 100%)
    df['Hedge_Index'] = ((df['Net Hedger'] - df['12M_Low']) / (df['12M_High'] - df['12M_Low'])) * 100
    return df

def generate_ict_signal(row):
    """Synthesizes the 'Zero Line' (Macro) with the 'Range Index' (Micro)."""
    net_pos = row['Net Hedger']
    hedge_idx = row['Hedge_Index']
    
    # Transcript: "When it's below the zero line... red... bearish."
    macro_program = "BUY PROGRAM (Above Zero)" if net_pos > 0 else "SELL PROGRAM (Below Zero)"
    macro_color = "green" if net_pos > 0 else "red"
    
    # Transcript: Commercials selling into rallies (Hedging against move)
    if hedge_idx <= 15:
        hedging_action = "📈 Aggressive Buying (Bullish Nodule)"
        context = "Commercials buying heavily relative to last year (Value)."
    elif hedge_idx >= 85:
        hedging_action = "📉 Aggressive Selling (Bearish Nodule)"
        context = "Commercials selling heavily relative to last year (Hedging)."
    else:
        hedging_action = "Passive / Neutral"
        context = "Commercials maintaining positions."
        
    return macro_program, macro_color, hedging_action, context

# --- ML HELPER FUNCTIONS ---
def calculate_rsi(series, period=14):
    delta = series.diff(); gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean(); rs = gain / loss
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
    w_df['Trend_Bullish'] = (w_df['Close'].rolling(window=20).mean() > w_df['Close'].rolling(window=50).mean()).astype(int)
    w_df['Target'] = (w_df['Close'].shift(-1) > w_df['Open'].shift(-1)).astype(int)
    return w_df.dropna()

def train_osok_model(ml_df):
    feature_cols = ['PD_Factor', 'Dist_Eq', 'RSI', 'Trend_Bullish']
    X = ml_df[feature_cols]; y = ml_df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model, accuracy_score(y_test, model.predict(X_test)), feature_cols

# --- WEEKLY ANALYSIS FUNCTIONS ---
def identify_weekly_profile(week_df):
    if week_df.empty: return {"Type": "Insufficient Data"}
    try: open_p = week_df.iloc[0]['Open'].item(); close_p = week_df.iloc[-1]['Close'].item()
    except: open_p = week_df.iloc[0]['Open']; close_p = week_df.iloc[-1]['Close']
    is_bullish = close_p > open_p
    trend = "Bullish" if is_bullish else "Bearish"
    high_date = week_df.loc[week_df['High'].idxmax(), 'Date']
    low_date = week_df.loc[week_df['Low'].idxmin(), 'Date']
    if hasattr(high_date, 'item'): high_date = high_date.item()
    if hasattr(low_date, 'item'): low_date = low_date.item()
    high_day = high_date.weekday(); low_day = low_date.weekday()
    
    profile = "Undefined"
    if is_bullish:
        if low_day == 1: profile = "Classic Tuesday Low"
        elif low_day == 0: profile = "Monday Low"
        elif low_day == 2: profile = "Wednesday Low / Reversal"
    else:
        if high_day == 1: profile = "Classic Tuesday High"
        elif high_day == 0: profile = "Monday High"
        elif high_day == 2: profile = "Wednesday High / Reversal"
        
    return {"Trend": trend, "Profile": profile, "High_Day": high_date.strftime("%A"), "Low_Day": low_date.strftime("%A")}

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

def identify_intraday_profile(df):
    if df.empty: return None
    midnight_bar = df[df['Datetime_NY'].dt.hour == 0]
    midnight_open = midnight_bar.iloc[0]['Open'] if not midnight_bar.empty else df.iloc[0]['Open']
    current_price = df.iloc[-1]['Close']
    is_bullish = current_price > midnight_open
    trend = "Bullish" if is_bullish else "Bearish"
    day_high = df['High'].max(); day_low = df['Low'].min()
    high_time = df.loc[df['High'].idxmax(), 'Datetime_NY'].time()
    low_time = df.loc[df['Low'].idxmin(), 'Datetime_NY'].time()
    
    profile = "Consolidation"
    judas_start = time(0, 0); judas_end = time(2, 0)
    
    if is_bullish:
        if judas_start <= low_time <= judas_end: profile = "London Normal (Buy)"
        elif low_time > judas_end: profile = "London Delayed (Buy)"
    else:
        if judas_start <= high_time <= judas_end: profile = "London Normal (Sell)"
        elif high_time > judas_end: profile = "London Delayed (Sell)"

    return {"Trend": trend, "Profile": profile, "Midnight_Open": midnight_open, "High_Time": high_time, "Low_Time": low_time}

# =========================================
# MAIN LOGIC BRANCHING
# =========================================

if analysis_mode == "Weekly Protocols (News & Logic)":
    st.sidebar.subheader("Protocol Settings")
    sim_mode = st.sidebar.checkbox("Simulate Jan 3-9, 2026 Context", value=True)
    st.title("📅 Weekly Profile Protocols")
    
    with st.spinner("Analyzing Calendar..."):
        if sim_mode:
            stories = [{"Title": "Oil prices steady", "Publisher": "Reuters", "Time": "2026-01-03 10:00"}]
            ff_df = pd.DataFrame() 
            current_date_ref = datetime(2026, 1, 5).date() 
        else:
            ff_df = scrape_forex_factory(week="this")
            stories = get_financial_news()
            current_date_ref = datetime.now().date()
            
    protocol, clustering, red_folder_days, exo_found = determine_weekly_protocol(ff_df, stories, manual_override_date=current_date_ref)
    
    col_stat1, col_stat2 = st.columns(2)
    col_stat1.metric("Current Day", current_date_ref.strftime("%A, %b %d"))
    col_stat2.metric("News Clustering", clustering)
    st.divider()
    
    action_color = THEME['bearish'] if "Avoid" in protocol['Action'] else THEME['bullish']
    st.markdown(f"""
    <div style="background-color: #1E252F; border-left: 5px solid {action_color}; padding: 20px; border-radius: 5px;">
        <h2 style="margin:0; color: white;">{protocol['Action']}</h2>
        <h4 style="margin:5px 0; color: #CCCCCC;">Profile: {protocol['Profile']}</h4>
        <p style="margin-top: 10px;">"{protocol['Logic']}"</p>
    </div>
    """, unsafe_allow_html=True)

elif analysis_mode == "Weekly Profiles":
    st.title(f"📊 Weekly Profile Identifier: {selected_asset_name}")
    lookback_weeks = st.sidebar.slider("Weeks to Display", 1, 20, 4)
    with st.spinner(f"Fetching data for {ticker_symbol}..."):
        df = get_data_weekly(ticker_symbol, 200)
    
    if df is not None:
        weeks = df.groupby('Week_Start')
        week_keys = sorted(list(weeks.groups.keys()), reverse=True)
        stats_data = []
        for w_start in week_keys[:100]:
            w_df = weeks.get_group(w_start)
            if len(w_df) >= 3:
                res = identify_weekly_profile(w_df)
                res['Week Start'] = w_start
                stats_data.append(res)
        stats_df = pd.DataFrame(stats_data)
        
        tab1, tab2 = st.tabs(["Analysis", "Seasonality"])
        with tab1:
            st.dataframe(stats_df.head(lookback_weeks), use_container_width=True)
        with tab2:
            avg_path = calculate_seasonal_path(weeks, 100)
            if avg_path is not None:
                fig = px.line(avg_path, x="DayName", y="PctChange", markers=True, title="Average Weekly Path")
                st.plotly_chart(apply_theme(fig), use_container_width=True)

elif analysis_mode == "Intraday Profiles":
    st.title(f"⏱️ Intraday Profile: {selected_asset_name}")
    target_date = st.sidebar.date_input("Select Trading Day", datetime.now().date())
    df_intra = get_data_intraday(ticker_symbol, target_date)
    
    if df_intra is not None and not df_intra.empty:
        res = identify_intraday_profile(df_intra)
        c1, c2, c3 = st.columns(3)
        c1.metric("Trend", res['Trend'])
        c2.metric("Profile", res['Profile'])
        c3.metric("Midnight Open", f"{res['Midnight_Open']:.5f}")
        
        fig = go.Figure(data=[go.Candlestick(x=df_intra['Datetime_NY'], open=df_intra['Open'], high=df_intra['High'], low=df_intra['Low'], close=df_intra['Close'])])
        fig.add_hline(y=res['Midnight_Open'], line_dash="dot", line_color="white")
        st.plotly_chart(apply_theme(fig), use_container_width=True)
    else: st.error("No Intraday data found.")

elif analysis_mode == "One Shot One Kill (OSOK)":
    st.title("🎯 One Shot One Kill (OSOK) + ML")
    st.markdown("### 20-Week IPDA Range & Institutional Sponsorship")
    st.caption("Using the transcript logic: Blending Seasonal Tendency, COT, and Weekly Profiles.")

    df_full = get_data_weekly(ticker_symbol, weeks=300)
    if df_full is not None:
        df_weekly = df_full.sort_values('Date')
        last_20 = df_weekly.iloc[-21:-1]
        current_week = df_weekly.iloc[-1]
        
        ipda_high = last_20['High'].max(); ipda_low = last_20['Low'].min()
        equilibrium = (ipda_high + ipda_low) / 2
        in_premium = current_week['Close'] > equilibrium
        
        col1, col2, col3 = st.columns(3)
        col1.metric("20-Week High (Liquidity)", f"{ipda_high:.4f}")
        col2.metric("20-Week Low (Liquidity)", f"{ipda_low:.4f}")
        col3.metric("Current State", "PREMIUM (Sell)" if in_premium else "DISCOUNT (Buy)", delta="-Short" if in_premium else "+Long", delta_color="inverse")
        
        st.progress((current_week['Close'] - ipda_low) / (ipda_high - ipda_low))
        st.caption("Position within 20-Week Range (0% = Low, 100% = High)")
        
        st.divider()
        st.subheader("OSOK Procedure (Transcript Checklist)")
        st.checkbox("1. Macro/Seasonal Conditions (Jan/March Turnover)", value=True)
        st.checkbox("2. COT Commercial Hedging Program Aligning?", value=False, help="Check 'COT Quant Terminal' tab")
        st.checkbox(f"3. {'Premium' if in_premium else 'Discount'} PD Array Matrix Identified?", value=True)
        st.checkbox("4. Monday/Tuesday High/Low Formed?", value=False)
        
        st.divider()
        st.subheader("🤖 ML Bias Predictor")
        ml_df = prepare_osok_ml_data(df_full)
        if len(ml_df) > 50:
            model, acc, _ = train_osok_model(ml_df)
            last_feat = ml_df.iloc[[-1]][['PD_Factor', 'Dist_Eq', 'RSI', 'Trend_Bullish']]
            prob = model.predict_proba(last_feat)[0]
            st.metric("Model Confidence (Next Week Close > Open)", f"{prob[1]*100:.1f}%", f"Accuracy: {acc*100:.1f}%")
        else: st.warning("Not enough data for ML.")

elif analysis_mode == "COT Quant Terminal":
    st.title("⚡ COT Quant Terminal")
    st.markdown("Automated **Commercial Hedging Program** detection using CFTC Data.")
    
    config = COT_ASSET_CONFIG[selected_cot_asset]
    spec_label, hedge_label = config['labels']

    @st.cache_data(ttl=3600)
    def fetch_multi_year(start_y, end_y, r_type):
        master_df = pd.DataFrame()
        for y in range(start_y, end_y + 1):
            df, err = load_cot_data(y, r_type)
            if df is not None: master_df = pd.concat([master_df, df])
        return master_df

    with st.spinner(f"Loading COT data for {selected_cot_asset}..."):
        df_raw = fetch_multi_year(cot_start_year, current_year, config['report_type'])

    if not df_raw.empty:
        keywords = config['keywords']
        mask = df_raw['market'].astype(str).apply(lambda x: all(k.lower() in x.lower() for k in keywords))
        df_asset = df_raw[mask].copy().sort_values('date')

        if not df_asset.empty:
            df_asset = calculate_hedging_ranges(df_asset)
            latest = df_asset.iloc[-1]; prev = df_asset.iloc[-2]
            macro_prog, macro_col, hedge_act, hedge_ctx = generate_ict_signal(latest)
            
            # DASHBOARD
            st.markdown(f"### 🧬 ICT Hedging Program Analysis ({hedge_label})")
            
            s1, s2 = st.columns([1, 2])
            with s1:
                st.markdown(f"**Macro Bias (Zero Line)**")
                st.markdown(f":{macro_col}[**{macro_prog}**]")
            with s2:
                st.markdown(f"**Hedging Activity (12-Month Range)**")
                st.markdown(f"**{hedge_act}**")
                st.caption(hedge_ctx)
                
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Commercial Net", f"{int(latest['Net Hedger']):,}", f"{int(latest['Net Hedger'] - prev['Net Hedger']):,}")
            m2.metric("12-Month High", f"{int(latest['12M_High']):,}")
            m3.metric("12-Month Low", f"{int(latest['12M_Low']):,}")
            m4.metric("Range Index", f"{latest['Hedge_Index']:.1f}%")

            # CHARTS
            tab1, tab2 = st.tabs(["📊 Commercial Hedging Program", "🦋 Open Interest Structure"])
            
            with tab1:
                st.markdown("**Instructions:** Focus on the Red Line (Commercials). The Shaded Area represents the 12-Month Range (Buying/Selling Program).")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df_asset['date'], y=df_asset['12M_High'], mode='lines', line=dict(width=0), showlegend=False))
                fig1.add_trace(go.Scatter(x=df_asset['date'], y=df_asset['12M_Low'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', name='12M Range'))
                fig1.add_trace(go.Scatter(x=df_asset['date'], y=df_asset['Net Hedger'], name=f'Commercials ({hedge_label})', line=dict(color='#FF0000', width=3)))
                fig1.add_hline(y=0, line_dash="solid", line_color="white", annotation_text="Zero Line")
                st.plotly_chart(apply_theme(fig1), use_container_width=True)
                
            with tab2:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=df_asset['date'], y=df_asset['hedge_long'], name=f"{hedge_label} Longs", marker_color='rgba(255, 0, 0, 0.6)'))
                fig2.add_trace(go.Bar(x=df_asset['date'], y=-df_asset['hedge_short'], name=f"{hedge_label} Shorts", marker_color='rgba(255, 0, 0, 0.3)'))
                fig2.add_trace(go.Scatter(x=df_asset['date'], y=df_asset['Net Hedger'], name="Net Position", line=dict(color='white', width=2, dash='dot')))
                fig2.update_layout(barmode='overlay')
                st.plotly_chart(apply_theme(fig2), use_container_width=True)
        else: st.warning(f"No data found for {selected_cot_asset}. Check keywords.")
    else: st.error("Could not fetch COT data.")

# --- MARKET SCANNER ---
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Market Scanner")
if st.sidebar.button("Scan All Assets"):
    st.info("Scanner functionality limited in this demo view.")
