import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, time
import pytz

# ML Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ICT Profiles Analyzer + ML",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# --- CUSTOM CSS FOR SLEEK PROFESSIONAL LOOK ---
st.markdown("""
<style>
    /* Main Background and Text */
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
        color: #e6edf3;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Custom Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #21262d;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        transition: transform 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="metric-container"]:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    
    [data-testid="stMetricLabel"] {
        color: #8b949e;
        font-size: 14px;
    }
    
    [data-testid="stMetricValue"] {
        color: #e6edf3;
        font-weight: 600;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid #30363d;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0e1117;
        border-radius: 4px 4px 0px 0px;
        color: #8b949e;
    }

    .stTabs [aria-selected="true"] {
        background-color: #21262d;
        color: #58a6ff;
        border-bottom: 2px solid #58a6ff;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Adjust top padding */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR SETTINGS ---
st.sidebar.title("⚙️ Configuration")

# 1. Analysis Mode
analysis_mode = st.sidebar.radio(
    "Analysis Mode", 
    ["Weekly Profiles", "Intraday Profiles", "One Shot One Kill (OSOK)"], 
    help="Weekly: Swing profiles. Intraday: London Protraction. OSOK: 20-Week Range + ML Prediction."
)

st.sidebar.markdown("---")

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

# --- SHARED HELPER FUNCTIONS ---

def get_data_weekly(ticker, weeks=52):
    """Fetches daily data for Weekly Analysis."""
    # Ensure enough data for ML training (at least 5 years)
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
    """Fetches intraday data. Interval can be 5m (Intraday) or 15m (OSOK)."""
    start_date = target_date - timedelta(days=2) 
    end_date = target_date + timedelta(days=2)
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        
        if data['Datetime'].dt.tz is None:
            data['Datetime'] = data['Datetime'].dt.tz_localize('UTC')
        
        data['Datetime_NY'] = data['Datetime'].dt.tz_convert('America/New_York')
        
        mask = data['Datetime_NY'].dt.date == target_date
        return data.loc[mask].copy()
    except Exception as e:
        st.error(f"Error fetching intraday data: {e}")
        return None

# --- ML HELPER FUNCTIONS ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_osok_ml_data(df):
    """
    Prepares features specifically based on OSOK concepts:
    - 20 Week Range Position (Premium/Discount)
    - Distance from Equilibrium
    - Recent Momentum
    """
    # Group by week to get weekly OHLC
    logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    w_df = df.groupby('Week_Start').agg(logic).sort_index()
    
    # Feature Engineering
    w_df['20W_High'] = w_df['High'].rolling(window=20).max()
    w_df['20W_Low'] = w_df['Low'].rolling(window=20).min()
    w_df['Equilibrium'] = (w_df['20W_High'] + w_df['20W_Low']) / 2
    
    # Feature 1: PD Factor (0 = Deep Discount, 1 = Deep Premium)
    w_df['PD_Factor'] = (w_df['Close'] - w_df['20W_Low']) / (w_df['20W_High'] - w_df['20W_Low'])
    
    # Feature 2: Distance from Equilibrium (%)
    w_df['Dist_Eq'] = (w_df['Close'] - w_df['Equilibrium']) / w_df['Equilibrium']
    
    # Feature 3: RSI (Momentum)
    w_df['RSI'] = calculate_rsi(w_df['Close'], 14)
    
    # Feature 4: Previous Week Return
    w_df['Prev_Ret'] = w_df['Close'].pct_change()
    
    # TARGET: Will NEXT week close higher than it opens? (Bullish Candle)
    # Shift(-1) looks at the future
    w_df['Target'] = (w_df['Close'].shift(-1) > w_df['Open'].shift(-1)).astype(int)
    
    w_df = w_df.dropna()
    return w_df

def train_osok_model(ml_df):
    feature_cols = ['PD_Factor', 'Dist_Eq', 'RSI', 'Prev_Ret']
    X = ml_df[feature_cols]
    y = ml_df['Target']
    
    # Split Data (Shuffle=False for time series to test on latest data)
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

# --- PREDICTION ENGINE (MARKOV CHAIN) ---
def predict_next_week(stats_df, current_profile):
    if stats_df.empty: return None
    df_sorted = stats_df.sort_values('Week Start', ascending=True).copy()
    df_sorted['Next_Profile'] = df_sorted['Profile'].shift(-1)
    transitions = df_sorted[df_sorted['Profile'] == current_profile]
    
    if transitions.empty or transitions['Next_Profile'].dropna().empty:
        return None
    
    counts = transitions['Next_Profile'].value_counts(normalize=True)
    return dict(sorted(counts.to_dict().items(), key=lambda item: item[1], reverse=True))

# --- INTRADAY ANALYSIS FUNCTIONS ---
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
    
    profile, desc = "Consolidation", "Choppy."
    
    if is_bullish:
        if judas_start <= low_time <= judas_end: profile, desc = "London Normal (Buy)", "Judas Swing Low (0-2 AM)."
        elif low_time > judas_end: profile, desc = "London Delayed (Buy)", "Low formed after 2 AM."
    else:
        if judas_start <= high_time <= judas_end: profile, desc = "London Normal (Sell)", "Judas Swing High (0-2 AM)."
        elif high_time > judas_end: profile, desc = "London Delayed (Sell)", "High formed after 2 AM."

    return {"Trend": trend, "Profile": profile, "Description": desc, "Midnight_Open": midnight_open, "High": day_high, "Low": day_low, "High_Time": high_time, "Low_Time": low_time}

# =========================================
# MAIN LOGIC BRANCHING
# =========================================

if analysis_mode == "Weekly Profiles":
    
    st.sidebar.subheader("Weekly Settings")
    lookback_weeks = st.sidebar.slider("Weeks to Display", 1, 20, 4)
    # Default high for better prediction accuracy
    stats_lookback = st.sidebar.slider("Stats Range (Prediction History)", 52, 300, 150)
    
    st.title(f"📊 Weekly Profile Identifier: {selected_asset_name}")
    
    with st.spinner(f"Fetching weekly data for {ticker_symbol}..."):
        # Fetching extra data handled by get_data_weekly optimization
        df = get_data_weekly(ticker_symbol, max(lookback_weeks, stats_lookback) + 3)

    if df is not None:
        weeks = df.groupby('Week_Start')
        week_keys = sorted(list(weeks.groups.keys()), reverse=True)
        week_keys = [k for k in week_keys if not pd.isna(k)]
        
        if not week_keys:
            st.error("No data.")
        else:
            stats_data = []
            # Calculate stats over the full requested range
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
                
                

                # --- PREDICTION SECTION ---
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
                # --------------------------
                
                # METRICS ROW
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Trend", analysis["Trend"], delta_color="normal" if analysis["Trend"]=="Bullish" else "inverse")
                c2.metric("Profile", analysis["Profile"])
                c3.metric("High Day", analysis["High_Day"])
                c4.metric("Low Day", analysis["Low_Day"])
                
                st.caption(f"**Logic:** {analysis['Description']}")
                
                fig = go.Figure(data=[go.Candlestick(x=curr_df['Date'], open=curr_df['Open'], high=curr_df['High'], low=curr_df['Low'], close=curr_df['Close'])])
                if prev_data:
                    fig.add_hline(y=prev_data['PWH'], line_dash="dash", line_color="orange", annotation_text="PWH")
                    fig.add_hline(y=prev_data['PWL'], line_dash="dash", line_color="orange", annotation_text="PWL", annotation_position="bottom right")
                
                fig.update_layout(title=f"Weekly Chart: {selected_asset_name}", template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
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
                        fig_seas.add_hline(y=0, line_dash="dot", line_color="white")
                        fig_seas.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_seas, use_container_width=True)
                    
                    st.divider()
                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.subheader("Day Probability")
                        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                        h_c = stats_df['High_Day'].value_counts().reindex(days, fill_value=0)
                        l_c = stats_df['Low_Day'].value_counts().reindex(days, fill_value=0)
                        df_c = pd.DataFrame({"Day": days, "Highs": h_c.values, "Lows": l_c.values}).melt(id_vars="Day", var_name="Type", value_name="Count")
                        fig_d = px.bar(df_c, x="Day", y="Count", color="Type", barmode="group", color_discrete_map={"Highs": "#EF553B", "Lows": "#00CC96"})
                        fig_d.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_d, use_container_width=True)
                    with c_b:
                        st.subheader("Profile Distribution")
                        profile_counts = stats_df['Profile'].value_counts().reset_index()
                        profile_counts.columns = ['Profile', 'Count'] 
                        fig_p = px.pie(profile_counts, names='Profile', values='Count', hole=0.4)
                        fig_p.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
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
        
        # PDH/PDL
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
        fig.add_trace(go.Candlestick(x=df_intra['Datetime_NY'], open=df_intra['Open'], high=df_intra['High'], low=df_intra['Low'], close=df_intra['Close'], name="Price (5m)"))
        fig.add_hline(y=res['Midnight_Open'], line_dash="dot", line_color="white", annotation_text="Midnight Open")
        
        base_dt = pd.to_datetime(target_date).tz_localize('America/New_York') if df_intra['Datetime_NY'].iloc[0].tzinfo else pd.to_datetime(target_date)
        def to_ms(h, m): return base_dt.replace(hour=h, minute=m).timestamp() * 1000

        fig.add_vline(x=to_ms(0,0), line_dash="solid", line_color="gray", annotation_text="00:00 NY")
        fig.add_vline(x=to_ms(2,0), line_dash="solid", line_color="gray", annotation_text="02:00 NY")
        
        if show_killzones:
            fig.add_vrect(x0=to_ms(2,0), x1=to_ms(5,0), fillcolor="green", opacity=0.07, annotation_text="London KZ", annotation_position="top left")
            fig.add_vrect(x0=to_ms(7,0), x1=to_ms(10,0), fillcolor="orange", opacity=0.07, annotation_text="NY KZ", annotation_position="top left")

        if prev_day_stats:
            fig.add_hline(y=prev_day_stats['PDH'], line_dash="dash", line_color="#EF553B", annotation_text="PDH")
            fig.add_hline(y=prev_day_stats['PDL'], line_dash="dash", line_color="#00CC96", annotation_text="PDL", annotation_position="bottom right")

        fig.update_layout(title=f"Intraday Chart (NY Time) - {target_date}", template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
    else: st.error("No Intraday data found.")

elif analysis_mode == "One Shot One Kill (OSOK)":
    
    st.sidebar.subheader("OSOK Settings")
    st.title("🎯 One Shot One Kill (OSOK) + ML Model")
    st.markdown("Identifies the **20-Week Dealing Range** and uses **Machine Learning** to predict the bias.")

    with st.spinner("Analyzing 20-Week IPDA Range..."):
        # Fetch lots of data for ML training
        df_full = get_data_weekly(ticker_symbol, weeks=300)
    
    if df_full is not None:
        # Separate data for OSOK Visuals
        df_weekly = df_full.sort_values('Date')
        last_20 = df_weekly.iloc[-21:-1]
        current_week = df_weekly.iloc[-1]
        
        ipda_high = last_20['High'].max()
        ipda_low = last_20['Low'].min()
        ipda_range = ipda_high - ipda_low
        equilibrium = (ipda_high + ipda_low) / 2
        
        current_close = current_week['Close']
        in_premium = current_close > equilibrium
        
        # Display Metrics in Cards
        m1, m2, m3 = st.columns(3)
        m1.metric("20-Week High (Liquidity)", f"{ipda_high:.2f}")
        m2.metric("20-Week Low (Liquidity)", f"{ipda_low:.2f}")
        m3.metric("Current State", "PREMIUM (Sell)" if in_premium else "DISCOUNT (Buy)", 
                  delta="-Sell" if in_premium else "+Buy", delta_color="inverse")
        
        st.markdown("### Range Position")
        st.progress((current_close - ipda_low) / ipda_range)
        st.caption(f"Price is at {((current_close - ipda_low) / ipda_range)*100:.1f}% of the 20-week range.")

        # --- ML MODEL INTEGRATION ---
        st.divider()
        st.subheader("🤖 OSOK ML Probability Engine")
        
        with st.expander("View ML Model Logic"):
            st.markdown("""
            **How this ML Model works:**
            1. **Feature Extraction:** It calculates where price is within the 20-week range (Premium/Discount factor).
            2. **Momentum:** It checks RSI and recent weekly returns.
            3. **Target:** It trains a Random Forest to predict if the *Next Week* closes higher than it opens.
            """)
        
        # Prepare Data
        ml_df = prepare_osok_ml_data(df_full)
        
        if len(ml_df) > 50:
            # Train Model
            model, acc, feats = train_osok_model(ml_df)
            
            # Predict for Current/Next Week
            # We take the latest known row to predict the future
            latest_features = ml_df.iloc[[-1]][feats]
            pred_prob = model.predict_proba(latest_features)[0] # [Prob Bearish, Prob Bullish]
            
            ml_c1, ml_c2 = st.columns(2)
            
            with ml_c1:
                st.metric("Model Backtest Accuracy", f"{acc*100:.1f}%")
                bias_score = pred_prob[1] # Probability of Bullish
                
                if bias_score > 0.55:
                    st.success(f"**ML Bias: BULLISH ({bias_score*100:.1f}%)**")
                elif bias_score < 0.45:
                    st.error(f"**ML Bias: BEARISH ({(1-bias_score)*100:.1f}%)**")
                else:
                    st.warning(f"**ML Bias: NEUTRAL / CHOPPY**")
                    
            with ml_c2:
                # Feature Importance
                importances = pd.DataFrame({'Feature': feats, 'Importance': model.feature_importances_})
                fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', title="Factor Importance")
                fig_imp.update_layout(template="plotly_dark", height=200, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_imp, use_container_width=True)
                
        else:
            st.warning("Not enough historical data to train the OSOK ML model reliably.")

        st.divider()
        # --- END ML INTEGRATION ---

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
            fig_osok.add_trace(go.Candlestick(x=df_15m['Datetime_NY'], open=df_15m['Open'], high=df_15m['High'], low=df_15m['Low'], close=df_15m['Close'], name="Price (15m)"))
            
            view_high = df_15m['High'].max()
            view_low = df_15m['Low'].min()
            diff = view_high - view_low
            
            if in_premium:
                ote_62 = view_low + (diff * 0.62)
                ote_79 = view_low + (diff * 0.79)
                color_ote = "red"
                bias_text = "Bearish OTE Zone (Sell)"
            else:
                ote_62 = view_high - (diff * 0.62)
                ote_79 = view_high - (diff * 0.79)
                color_ote = "green"
                bias_text = "Bullish OTE Zone (Buy)"
            
            fig_osok.add_hrect(y0=ote_62, y1=ote_79, fillcolor=color_ote, opacity=0.1, annotation_text=bias_text, annotation_position="right")
            
            base_dt_osok = pd.to_datetime(target_date_osok).tz_localize('America/New_York') if df_15m['Datetime_NY'].iloc[0].tzinfo else pd.to_datetime(target_date_osok)
            def to_ms_osok(h, m): return base_dt_osok.replace(hour=h, minute=m).timestamp() * 1000
            
            fig_osok.add_vrect(x0=to_ms_osok(2,0), x1=to_ms_osok(5,0), fillcolor="green", opacity=0.07, annotation_text="London KZ", annotation_position="top left")
            fig_osok.add_vrect(x0=to_ms_osok(7,0), x1=to_ms_osok(10,0), fillcolor="orange", opacity=0.07, annotation_text="NY KZ", annotation_position="top left")

            fig_osok.update_layout(title=f"OSOK Execution (15m) - {target_date_osok}", template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_osok, use_container_width=True)
            
            st.info(f"**OSOK Checklist:** 1. Price in { 'Premium' if in_premium else 'Discount' } of 20-week range? ✅ | 2. Is it Mon/Tue/Wed? {'✅' if is_anchor_day else '❌'} | 3. Wait for price to hit the OTE Zone during a Kill Zone.")
            
        else:
            st.warning("No intraday data available for the selected date.")
    else:
        st.error("Could not fetch weekly data.")

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
    else:
        st.warning("No data found.")
