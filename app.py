import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ICT Weekly Profiles Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR SETTINGS ---
st.sidebar.title("Configuration")

# Asset Selection
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

# Lookback Period for the Table/View
lookback_weeks = st.sidebar.slider("Recent Weeks to Display", min_value=1, max_value=20, value=4)

# Lookback Period for Statistics
stats_lookback = st.sidebar.slider("Statistical Analysis Range (Weeks)", min_value=10, max_value=104, value=52, help="How far back to look for the probability stats.")

# --- HELPER FUNCTIONS ---

def get_data(ticker, weeks=52):
    """
    Fetches daily data from yfinance.
    """
    period_days = weeks * 7 + 14 # Buffer
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)
    
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    
    if data.empty:
        return None

    # --- FIX: Flatten MultiIndex Columns ---
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Reset index to make Date a column
    data = data.reset_index()
    
    # Ensure Date is datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Identify the start of the week (Monday) for grouping
    data['Week_Start'] = data['Date'].apply(lambda x: x - timedelta(days=x.weekday()))
    
    return data

def identify_profile(week_df):
    """
    Analyzes a single week's dataframe to identify the ICT Weekly Profile.
    """
    if week_df.empty:
        return {"Type": "Insufficient Data", "Desc": "No data available"}

    # Basic OHLC
    try:
        open_price = week_df.iloc[0]['Open'].item()
        close_price = week_df.iloc[-1]['Close'].item()
    except AttributeError:
        open_price = week_df.iloc[0]['Open']
        close_price = week_df.iloc[-1]['Close']

    high_price = week_df['High'].max()
    low_price = week_df['Low'].min()
    
    # Trend
    is_bullish = close_price > open_price
    trend = "Bullish" if is_bullish else "Bearish"
    
    # Identify Day of Week for High and Low
    high_date = week_df.loc[week_df['High'].idxmax(), 'Date']
    low_date = week_df.loc[week_df['Low'].idxmin(), 'Date']
    
    if hasattr(high_date, 'item'): high_date = high_date.item()
    if hasattr(low_date, 'item'): low_date = low_date.item()
    
    high_day_num = high_date.weekday() # 0=Mon, 1=Tue...
    low_day_num = low_date.weekday()
    
    profile_name = "Undefined"
    description = "Pattern does not fit strict classic definitions."

    # --- LOGIC RULES ---
    if is_bullish:
        if low_day_num == 1: # Tuesday
            profile_name = "Classic Tuesday Low"
            description = "Low formed on Tuesday, followed by expansion higher."
        elif low_day_num == 0: # Monday
            profile_name = "Monday Low"
            description = "Low formed on Monday. Often leads to a steady expansion."
        elif low_day_num == 2: # Wednesday
            profile_name = "Wednesday Low / Reversal"
            description = "Market manipulated Mon-Tue, Low formed Wednesday."
        elif low_day_num == 3: # Thursday
            profile_name = "Consolidation Thursday Reversal"
            description = "Consolidation Mon-Wed, Stop hunt Low on Thursday, then Reversal."
        elif low_day_num == 4: # Friday
            profile_name = "Seek & Destroy / Friday Low"
            description = "Choppy week, Low formed late on Friday."
            
    else: 
        if high_day_num == 1: # Tuesday
            profile_name = "Classic Tuesday High"
            description = "High formed on Tuesday, followed by expansion lower."
        elif high_day_num == 0: # Monday
            profile_name = "Monday High"
            description = "High formed on Monday. Often leads to a steady decline."
        elif high_day_num == 2: # Wednesday
            profile_name = "Wednesday High / Reversal"
            description = "Market manipulated Mon-Tue, High formed Wednesday."
        elif high_day_num == 3: # Thursday
            profile_name = "Consolidation Thursday Reversal"
            description = "Consolidation Mon-Wed, Stop hunt High on Thursday, then Reversal."
        elif high_day_num == 4: # Friday
            profile_name = "Seek & Destroy / Friday High"
            description = "Choppy week, High formed late on Friday."

    return {
        "Trend": trend,
        "Profile": profile_name,
        "Description": description,
        "Weekly_High": high_price,
        "Weekly_Low": low_price,
        "High_Day": high_date.strftime("%A"),
        "Low_Day": low_date.strftime("%A"),
        "High_Day_Num": high_day_num,
        "Low_Day_Num": low_day_num
    }

# --- MAIN APP LOGIC ---

st.title(f"📊 ICT Weekly Profile Identifier: {selected_asset_name}")
st.markdown(f"Analysis based on **{ticker_symbol}** price action.")

# 1. Load Data (Fetch larger range for stats)
with st.spinner('Fetching market data...'):
    # We fetch the MAX of stats_lookback or lookback_weeks to ensure we have enough data
    fetch_weeks = max(lookback_weeks, stats_lookback) + 2
    df = get_data(ticker_symbol, weeks=fetch_weeks)

if df is not None:
    # Group data by Week
    weeks = df.groupby('Week_Start')
    week_keys = list(weeks.groups.keys())
    week_keys.sort(reverse=True)
    
    # Filter valid keys
    week_keys = [k for k in week_keys if not pd.isna(k)]
    
    if not week_keys:
        st.error("No valid weekly data found.")
        st.stop()

    # --- PROCESS ALL DATA FOR STATISTICS ---
    stats_data = []
    # Analyze all available weeks within the stats_lookback range
    for w_start in week_keys[:stats_lookback]:
        w_df = weeks.get_group(w_start)
        # Only analyze if the week has at least 3 days of data to be significant
        if len(w_df) >= 3:
            res = identify_profile(w_df)
            res['Week Start'] = w_start
            stats_data.append(res)
    
    stats_df = pd.DataFrame(stats_data)

    # --- TABS LAYOUT ---
    tab1, tab2 = st.tabs(["🔎 Current Analysis", "📈 Statistical Probability"])

    # ==========================
    # TAB 1: CURRENT ANALYSIS
    # ==========================
    with tab1:
        selected_week_start = st.selectbox(
            "Select Week to View Chart", 
            week_keys[:lookback_weeks],
            format_func=lambda x: f"Week of {x.strftime('%Y-%m-%d')}"
        )
        
        current_week_df = weeks.get_group(selected_week_start).copy()
        analysis = identify_profile(current_week_df)
        
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Weekly Trend", analysis["Trend"], delta_color="normal" if analysis["Trend"]=="Bullish" else "inverse")
        c2.metric("Detected Profile", analysis["Profile"])
        c3.metric("High Formed On", analysis["High_Day"])
        c4.metric("Low Formed On", analysis["Low_Day"])
        
        st.info(f"**Logic:** {analysis['Description']}")

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=current_week_df['Date'], open=current_week_df['Open'],
            high=current_week_df['High'], low=current_week_df['Low'],
            close=current_week_df['Close'], name='Price'
        ))

        # Annotations
        max_idx = current_week_df['High'].idxmax()
        min_idx = current_week_df['Low'].idxmin()
        
        fig.add_annotation(x=current_week_df.loc[max_idx, 'Date'], y=current_week_df.loc[max_idx, 'High'],
                           text="High", showarrow=True, arrowhead=1, ay=-40)
        fig.add_annotation(x=current_week_df.loc[min_idx, 'Date'], y=current_week_df.loc[min_idx, 'Low'],
                           text="Low", showarrow=True, arrowhead=1, ay=40)

        fig.update_layout(
            title=f"{selected_asset_name} - Week of {selected_week_start.strftime('%Y-%m-%d')}",
            xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False,
            height=500, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Recent History Table
        st.subheader("Recent History")
        # Reuse stats_df but filter for the view limit
        display_cols = ['Week Start', 'Trend', 'Profile', 'High_Day', 'Low_Day']
        st.dataframe(stats_df[display_cols].head(lookback_weeks), use_container_width=True)

    # ==========================
    # TAB 2: STATISTICAL PROBABILITY
    # ==========================
    with tab2:
        st.markdown(f"### Statistical Analysis (Last {len(stats_df)} Weeks)")
        
        if not stats_df.empty:
            # 1. Day of Week Probability (Highs vs Lows)
            st.subheader("Day of Week Probability")
            st.caption("How often does the High or Low form on a specific day?")
            
            # Prepare data for plotting
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            
            # Count Highs per day
            high_counts = stats_df['High_Day'].value_counts().reindex(days_order, fill_value=0)
            # Count Lows per day
            low_counts = stats_df['Low_Day'].value_counts().reindex(days_order, fill_value=0)
            
            # Combine into a format for Plotly Express
            counts_df = pd.DataFrame({
                "Day": days_order,
                "High Formation": high_counts.values,
                "Low Formation": low_counts.values
            })
            
            # Melting for Grouped Bar Chart
            melted_counts = counts_df.melt(id_vars="Day", var_name="Type", value_name="Count")
            
            fig_days = px.bar(
                melted_counts, x="Day", y="Count", color="Type", barmode="group",
                title=f"High/Low Formation Frequency ({len(stats_df)} weeks)",
                color_discrete_map={"High Formation": "#EF553B", "Low Formation": "#00CC96"}
            )
            fig_days.update_layout(template="plotly_dark")
            st.plotly_chart(fig_days, use_container_width=True)
            
            # 2. Profile Distribution
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Most Common Profiles")
                profile_counts = stats_df['Profile'].value_counts().reset_index()
                profile_counts.columns = ['Profile', 'Count']
                
                fig_prof = px.pie(profile_counts, names='Profile', values='Count', hole=0.4,
                                  title="Profile Type Distribution")
                fig_prof.update_layout(template="plotly_dark")
                st.plotly_chart(fig_prof, use_container_width=True)
                
            with col_b:
                st.subheader("Trend Bias")
                trend_counts = stats_df['Trend'].value_counts().reset_index()
                trend_counts.columns = ['Trend', 'Count']
                
                fig_trend = px.bar(trend_counts, x='Trend', y='Count', color='Trend',
                                   color_discrete_map={"Bullish": "#00CC96", "Bearish": "#EF553B"},
                                   title="Weekly Trend Direction")
                fig_trend.update_layout(template="plotly_dark")
                st.plotly_chart(fig_trend, use_container_width=True)

        else:
            st.warning("Not enough data to generate statistics.")

else:
    st.error("Could not fetch data. Please check your connection or try again later.")

# --- FOOTER ---
st.markdown("---")
st.caption("Statistical data is calculated based on the 'Statistical Analysis Range' slider in the sidebar.")
