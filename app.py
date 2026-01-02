import streamlit as st
import yfinance as yf
import pandas as pd
import cot_reports as cot  # NEW DEPENDENCY
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, time
import pytz

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ICT Profiles Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. CONFIGURATION & MAPPINGS ---
st.sidebar.title("Configuration")

# Analysis Mode
analysis_mode = st.sidebar.radio(
    "Analysis Mode", 
    ["Weekly Profiles", "Intraday Profiles", "One Shot One Kill (OSOK)", "Institutional COT Data"], 
    help="Weekly: Swing profiles. Intraday: London Protraction. OSOK: 20-Week IPDA Range. COT: Smart Money Positioning."
)

# Fallback Toggle
use_etf = st.sidebar.checkbox("Use ETF Tickers (More Stable)", value=False, 
    help="Check this if Futures data (ES=F, GC=F) fails to load. ETFs (SPY, GLD) are more reliable on the free API.")

# Asset Selection
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

# --- COT CONFIGURATION ---
# Maps the app's selected asset to COT Report Keywords
COT_MAPPING = {
    "Gold": {
        "keywords": ["GOLD", "COMMODITY EXCHANGE"],
        "report_type": "disaggregated_fut",
        "labels": ("Large Speculators", "Commercials (Smart Money)")
    },
    "S&P 500": {
        "keywords": ["E-MINI S&P 500", "CHICAGO MERCANTILE EXCHANGE"],
        "report_type": "traders_in_financial_futures_fut",
        "labels": ("Leveraged Funds", "Asset Managers/Commercials")
    },
    "Nasdaq 100": {
        "keywords": ["E-MINI NASDAQ-100", "CHICAGO MERCANTILE EXCHANGE"],
        "report_type": "traders_in_financial_futures_fut",
        "labels": ("Leveraged Funds", "Asset Managers/Commercials")
    },
    "EUR/USD": {
        "keywords": ["EURO FX", "CHICAGO MERCANTILE EXCHANGE"],
        "report_type": "traders_in_financial_futures_fut",
        "labels": ("Leveraged Funds", "Asset Managers/Commercials")
    },
    "GBP/USD": {
        "keywords": ["BRITISH POUND", "CHICAGO MERCANTILE EXCHANGE"],
        "report_type": "traders_in_financial_futures_fut",
        "labels": ("Leveraged Funds", "Asset Managers/Commercials")
    },
    "Bitcoin": {
        "keywords": ["BITCOIN", "CHICAGO MERCANTILE EXCHANGE"],
        "report_type": "traders_in_financial_futures_fut",
        "labels": ("Leveraged Funds", "Commercials")
    }
}

# Helper to find the right COT config based on selected name
def get_cot_config(selected_name):
    for key in COT_MAPPING:
        if key in selected_name:
            return COT_MAPPING[key]
    return None

# --- SHARED HELPER FUNCTIONS (PRICE DATA) ---

def get_data_weekly(ticker, weeks=52):
    min_history_weeks = 260 
    fetch_weeks = max(weeks, min_history_weeks)
    period_days = fetch_weeks * 7 + 21
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
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def get_data_intraday(ticker, target_date, interval="5m"):
    start_date = target_date - timedelta(days=2) 
    end_date = target_date + timedelta(days=2)
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        if data['Datetime'].dt.tz is None: data['Datetime'] = data['Datetime'].dt.tz_localize('UTC')
        data['Datetime_NY'] = data['Datetime'].dt.tz_convert('America/New_York')
        mask = data['Datetime_NY'].dt.date == target_date
        return data.loc[mask].copy()
    except Exception: return None

# --- COT HELPER FUNCTIONS ---

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
        col_map['hedge_long'] = get_col(['asset', 'mgr', 'long'])
        col_map['hedge_short'] = get_col(['asset', 'mgr', 'short'])

    final_map = {v: k for k, v in col_map.items() if v}
    df = df.rename(columns=final_map)
    
    for c in ['spec_long', 'spec_short', 'hedge_long', 'hedge_short']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
    return df

@st.cache_data(ttl=3600)
def fetch_cot_data(start_year, end_year, report_type):
    master_df = pd.DataFrame()
    for year in range(start_year, end_year + 1):
        try:
            df = cot.cot_year(year=year, cot_report_type=report_type)
            if df is not None and not df.empty:
                df = clean_headers(df)
                df = map_columns(df, report_type)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.dropna(subset=['date'])
                    master_df = pd.concat([master_df, df])
        except: continue
    return master_df

def calculate_z_score(series, window=52):
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()
    return (series - roll_mean) / roll_std

# --- ANALYSIS FUNCTIONS (OLD) ---
def identify_weekly_profile(week_df):
    if week_df.empty: return {"Type": "Insufficient Data"}
    try: open_p, close_p = week_df.iloc[0]['Open'].item(), week_df.iloc[-1]['Close'].item()
    except: open_p, close_p = week_df.iloc[0]['Open'], week_df.iloc[-1]['Close']
    
    is_bullish = close_p > open_p
    trend = "Bullish" if is_bullish else "Bearish"
    high_date, low_date = week_df.loc[week_df['High'].idxmax(), 'Date'], week_df.loc[week_df['Low'].idxmin(), 'Date']
    if hasattr(high_date, 'item'): high_date = high_date.item()
    if hasattr(low_date, 'item'): low_date = low_date.item()
    high_day, low_day = high_date.weekday(), low_date.weekday()
    
    profile, desc = "Undefined", "N/A"
    if is_bullish:
        if low_day == 1: profile, desc = "Classic Tuesday Low", "Low on Tue, expansion higher."
        elif low_day == 0: profile, desc = "Monday Low", "Low on Mon, steady expansion."
        elif low_day == 2: profile, desc = "Wednesday Low / Reversal", "Low on Wed."
        elif low_day == 3: profile, desc = "Consolidation Thu Reversal", "Stop hunt Low on Thu, then Reversal."
        elif low_day == 4: profile, desc = "Seek & Destroy / Fri Low", "Choppy, Low late Friday."
    else:
        if high_day == 1: profile, desc = "Classic Tuesday High", "High on Tue, expansion lower."
        elif high_day == 0: profile, desc = "Monday High", "High on Mon, steady decline."
        elif high_day == 2: profile, desc = "Wednesday High / Reversal", "High on Wed."
        elif high_day == 3: profile, desc = "Consolidation Thu Reversal", "Stop hunt High on Thu, then Reversal."
        elif high_day == 4: profile, desc = "Seek & Destroy / Fri High", "Choppy, High late Friday."
        
    return {"Trend": trend, "Profile": profile, "Description": desc, "High_Day": high_date.strftime("%A"), "Low_Day": low_date.strftime("%A")}

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
    midnight_open = df.iloc[0]['Open'] if midnight_bar.empty else midnight_bar.iloc[0]['Open']
    judas_start, judas_end = time(0, 0), time(2, 0)
    current_price = df.iloc[-1]['Close']
    is_bullish = current_price > midnight_open
    trend = "Bullish" if is_bullish else "Bearish"
    
    high_time = df.loc[df['High'].idxmax(), 'Datetime_NY'].time()
    low_time = df.loc[df['Low'].idxmin(), 'Datetime_NY'].time()
    
    profile, desc = "Consolidation", "Choppy."
    if is_bullish:
        if judas_start <= low_time <= judas_end: profile, desc = "London Normal (Buy)", "Judas Swing Low (0-2 AM)."
        elif low_time > judas_end: profile, desc = "London Delayed (Buy)", "Low formed after 2 AM."
    else:
        if judas_start <= high_time <= judas_end: profile, desc = "London Normal (Sell)", "Judas Swing High (0-2 AM)."
        elif high_time > judas_end: profile, desc = "London Delayed (Sell)", "High formed after 2 AM."

    return {"Trend": trend, "Profile": profile, "Description": desc, "Midnight_Open": midnight_open, "High_Time": high_time, "Low_Time": low_time}

# =========================================
# MAIN LOGIC BRANCHING
# =========================================

if analysis_mode == "Weekly Profiles":
    st.sidebar.subheader("Weekly Settings")
    lookback_weeks = st.sidebar.slider("Weeks to Display", 1, 20, 4)
    stats_lookback = st.sidebar.slider("Stats Range", 52, 300, 150)
    
    st.title(f"📊 Weekly Profile: {selected_asset_name}")
    df = get_data_weekly(ticker_symbol, max(lookback_weeks, stats_lookback) + 3)

    if df is not None:
        weeks = df.groupby('Week_Start')
        week_keys = sorted(list(weeks.groups.keys()), reverse=True)
        if not week_keys: st.error("No data.")
        else:
            stats_data = []
            for w_start in week_keys[:stats_lookback]:
                w_df = weeks.get_group(w_start)
                if len(w_df) >= 3:
                    res = identify_weekly_profile(w_df)
                    res['Week Start'] = w_start
                    stats_data.append(res)
            stats_df = pd.DataFrame(stats_data)

            tab1, tab2 = st.tabs(["Current Analysis", "Stats"])
            with tab1:
                sel_week = st.selectbox("Select Week", week_keys[:lookback_weeks], format_func=lambda x: f"Week of {x.strftime('%Y-%m-%d')}")
                curr_df = weeks.get_group(sel_week).copy()
                analysis = identify_weekly_profile(curr_df)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Trend", analysis["Trend"], delta_color="normal" if analysis["Trend"]=="Bullish" else "inverse")
                c2.metric("Profile", analysis["Profile"])
                c3.metric("High Day", analysis["High_Day"])
                c4.metric("Low Day", analysis["Low_Day"])
                
                fig = go.Figure(data=[go.Candlestick(x=curr_df['Date'], open=curr_df['Open'], high=curr_df['High'], low=curr_df['Low'], close=curr_df['Close'])])
                fig.update_layout(title=f"Weekly Chart: {selected_asset_name}", template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                if not stats_df.empty:
                    avg_path = calculate_seasonal_path(weeks, stats_lookback)
                    if avg_path is not None:
                        fig_seas = px.line(avg_path, x="DayName", y="PctChange", markers=True, title="Average Weekly Path")
                        fig_seas.add_hline(y=0, line_dash="dot", line_color="white")
                        fig_seas.update_layout(template="plotly_dark")
                        st.plotly_chart(fig_seas, use_container_width=True)

elif analysis_mode == "Intraday Profiles":
    st.sidebar.subheader("Intraday Settings")
    today = datetime.now().date()
    target_date = st.sidebar.date_input("Select Trading Day", today)
    show_killzones = st.sidebar.checkbox("Show Kill Zones", value=True)
    
    st.title(f"⏱️ Intraday Profile: {selected_asset_name}")
    df_intra = get_data_intraday(ticker_symbol, target_date)
        
    if df_intra is not None and not df_intra.empty:
        res = identify_intraday_profile(df_intra)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trend", res['Trend'], delta_color="normal" if res['Trend']=="Bullish" else "inverse")
        c2.metric("Profile", res['Profile'])
        c3.metric("High Time", str(res['High_Time'])[:5])
        c4.metric("Low Time", str(res['Low_Time'])[:5])
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_intra['Datetime_NY'], open=df_intra['Open'], high=df_intra['High'], low=df_intra['Low'], close=df_intra['Close'], name="Price"))
        fig.add_hline(y=res['Midnight_Open'], line_dash="dot", line_color="white", annotation_text="Midnight Open")
        
        base_dt = pd.to_datetime(target_date).tz_localize('America/New_York') if df_intra['Datetime_NY'].iloc[0].tzinfo else pd.to_datetime(target_date)
        def to_ms(h, m): return base_dt.replace(hour=h, minute=m).timestamp() * 1000

        fig.add_vline(x=to_ms(0,0), line_dash="solid", line_color="gray", annotation_text="00:00")
        fig.add_vline(x=to_ms(2,0), line_dash="solid", line_color="gray", annotation_text="02:00")
        if show_killzones:
            fig.add_vrect(x0=to_ms(2,0), x1=to_ms(5,0), fillcolor="green", opacity=0.07, annotation_text="London")
            fig.add_vrect(x0=to_ms(7,0), x1=to_ms(10,0), fillcolor="orange", opacity=0.07, annotation_text="NY")

        fig.update_layout(title=f"Intraday Chart (NY Time)", template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else: st.error("No Intraday data found.")

elif analysis_mode == "One Shot One Kill (OSOK)":
    st.title("🎯 One Shot One Kill (OSOK)")
    with st.spinner("Analyzing 20-Week IPDA Range..."):
        df_weekly = get_data_weekly(ticker_symbol, weeks=25)
    
    if df_weekly is not None:
        df_weekly = df_weekly.sort_values('Date')
        last_20 = df_weekly.iloc[-21:-1]
        curr = df_weekly.iloc[-1]['Close']
        high, low = last_20['High'].max(), last_20['Low'].min()
        in_premium = curr > (high + low) / 2
        
        m1, m2, m3 = st.columns(3)
        m1.metric("20-Wk High", f"{high:.2f}")
        m2.metric("20-Wk Low", f"{low:.2f}")
        m3.metric("State", "PREMIUM (Sell)" if in_premium else "DISCOUNT (Buy)", delta="-Sell" if in_premium else "+Buy", delta_color="inverse")
        st.caption("Look for OTE setups in the direction of the state.")

elif analysis_mode == "Institutional COT Data":
    # --- COT ANALYSIS LOGIC ---
    cot_config = get_cot_config(selected_asset_name)
    
    if not cot_config:
        st.warning(f"COT Data not available/configured for {selected_asset_name}.")
    else:
        st.title(f"🏛️ Institutional COT Data: {selected_asset_name}")
        st.markdown("**Following the 'Smart Money' (Commercials) vs. Retail/Speculators.**")
        
        # Lookback Input
        current_year = datetime.now().year
        start_year = st.sidebar.number_input("Lookback Start Year", min_value=2015, max_value=current_year, value=2021)
        
        with st.spinner("Fetching CFTC Reports..."):
            df_raw = fetch_cot_data(start_year, current_year, cot_config['report_type'])
            
        if not df_raw.empty:
            # Filter for specific asset keywords
            keywords = cot_config['keywords']
            mask = df_raw['market'].astype(str).apply(lambda x: all(k.lower() in x.lower() for k in keywords))
            df_asset = df_raw[mask].copy().sort_values('date')
            
            if not df_asset.empty and all(c in df_asset.columns for c in ['spec_long', 'spec_short', 'hedge_long', 'hedge_short']):
                
                # Calculate Net Positions
                df_asset['Net Speculator'] = df_asset['spec_long'] - df_asset['spec_short']
                # Hedgers = Commercials
                df_asset['Net Commercial'] = df_asset['hedge_long'] - df_asset['hedge_short']
                df_asset['Comm Z-Score'] = calculate_z_score(df_asset['Net Commercial'])
                
                latest = df_asset.iloc[-1]
                prev = df_asset.iloc[-2]
                
                spec_label, comm_label = cot_config['labels']
                
                # --- METRICS ---
                col1, col2, col3, col4 = st.columns(4)
                
                # Commercials (Red Line focus)
                comm_net = int(latest['Net Commercial'])
                comm_delta = comm_net - int(prev['Net Commercial'])
                # Interpretation: Commercial Net Long > 0 is Bullish
                comm_signal = "BULLISH" if comm_net > 0 else "BEARISH"
                
                col1.metric(f"{comm_label} (Net)", f"{comm_net:,}", f"{comm_delta:,}", 
                            delta_color="normal" if comm_delta > 0 else "inverse")
                
                spec_net = int(latest['Net Speculator'])
                spec_delta = spec_net - int(prev['Net Speculator'])
                col2.metric(f"{spec_label} (Net)", f"{spec_net:,}", f"{spec_delta:,}")
                
                z_val = latest['Comm Z-Score']
                col3.metric("Smart Money Z-Score", f"{z_val:.2f}σ", "Extreme" if abs(z_val) > 2 else "Neutral")
                col4.metric("Bias Signal", comm_signal, delta_color="normal" if comm_signal=="BULLISH" else "inverse")
                
                st.info(f"""
                **Analysis:** Commercials are currently **{comm_signal}** (Net Position: {comm_net:,}). 
                If Z-Score is extreme (>2 or <-2), expect a potential market reversal.
                """)
                
                # --- CHARTS ---
                tab1, tab2, tab3 = st.tabs(["Net Trend (Smart Money)", "Commercial Structure", "Z-Score Extremes"])
                
                with tab1:
                    
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=df_asset['date'], y=df_asset['Net Commercial'], name=comm_label, line=dict(color='#FF4B4B', width=2)))
                    fig1.add_trace(go.Scatter(x=df_asset['date'], y=df_asset['Net Speculator'], name=spec_label, line=dict(color='#00C805', width=2)))
                    fig1.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig1.update_layout(title="Net Positioning: Smart Money (Red) vs Speculators (Green)", template="plotly_dark", height=500, hovermode="x unified")
                    st.plotly_chart(fig1, use_container_width=True)
                    
                with tab2:
                    
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(x=df_asset['date'], y=df_asset['hedge_long'], name=f"{comm_label} Longs", marker_color='#00C805'))
                    fig2.add_trace(go.Bar(x=df_asset['date'], y=-df_asset['hedge_short'], name=f"{comm_label} Shorts", marker_color='#FF4B4B'))
                    fig2.update_layout(title=f"{comm_label} Structure (Butterfly)", template="plotly_dark", barmode='overlay', height=500, hovermode="x unified")
                    st.caption("Visualizing the total Long vs Short exposure of Smart Money.")
                    st.plotly_chart(fig2, use_container_width=True)
                    
                with tab3:
                    fig3 = go.Figure()
                    colors = ['red' if val > 2 or val < -2 else 'gray' for val in df_asset['Comm Z-Score']]
                    fig3.add_trace(go.Bar(x=df_asset['date'], y=df_asset['Comm Z-Score'], marker_color=colors, name="Z-Score"))
                    fig3.add_hline(y=2, line_dash="dot", line_color="red")
                    fig3.add_hline(y=-2, line_dash="dot", line_color="red")
                    fig3.update_layout(title="Smart Money Extremes (Z-Score > 2 indicates Reversal risk)", template="plotly_dark", height=500)
                    st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("Data found but columns missing (check report type mapping).")
        else:
            st.error("No COT data found for this asset in the selected timeframe.")

# --- MARKET SCANNER ---
if st.sidebar.button("Scan All Assets"):
    st.sidebar.info("Scanner only covers Price Action modes, not COT.")
