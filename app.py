import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
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

# Lookback Period
lookback_weeks = st.sidebar.slider("Weeks to Analyze", min_value=1, max_value=12, value=4)

# --- HELPER FUNCTIONS ---

def get_data(ticker, weeks=12):
    """
    Fetches daily data from yfinance for the specified number of weeks.
    Adds 'Week_Start' column to group data by week.
    """
    # Fetch enough days to cover the weeks (approx 7 days * weeks + buffer)
    period_days = weeks * 10 
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)
    
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    
    if data.empty:
        return None

    # --- FIX: Flatten MultiIndex Columns ---
    # Newer yfinance versions return columns like ('Open', 'GC=F'). We just want 'Open'.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    # ---------------------------------------

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
    Returns a dictionary with profile details.
    """
    if week_df.empty:
        return {"Type": "Insufficient Data", "Desc": "No data available"}

    # Basic OHLC for the week
    # .item() is used to ensure we get a Python scalar, not a pandas Series, to avoid ambiguity errors
    try:
        open_price = week_df.iloc[0]['Open'].item()
        close_price = week_df.iloc[-1]['Close'].item()
    except AttributeError:
        # Fallback if .item() isn't needed/available (older pandas versions)
        open_price = week_df.iloc[0]['Open']
        close_price = week_df.iloc[-1]['Close']

    high_price = week_df['High'].max()
    low_price = week_df['Low'].min()
    
    # Determine Trend
    is_bullish = close_price > open_price
    trend = "Bullish" if is_bullish else "Bearish"
    
    # Identify Day of Week for High and Low
    # idxmax returns the index, we need the row
    high_date = week_df.loc[week_df['High'].idxmax(), 'Date']
    low_date = week_df.loc[week_df['Low'].idxmin(), 'Date']
    
    # Ensure we handle Scalar values if they are wrapped
    if hasattr(high_date, 'item'): high_date = high_date.item()
    if hasattr(low_date, 'item'): low_date = low_date.item()
    
    high_day = high_date.weekday() # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
    low_day = low_date.weekday()
    
    profile_name = "Undefined / Consolidation"
    description = "Pattern does not fit strict classic definitions."

    # --- LOGIC RULES ---
    # Based on ICT concepts: The timing of the Low (for Bullish) or High (for Bearish) 
    # determines the profile type.

    if is_bullish:
        # BULLISH PROFILES
        if low_day == 1: # Tuesday
            profile_name = "Classic Tuesday Low"
            description = "Low formed on Tuesday, followed by expansion higher."
        elif low_day == 0: # Monday
            # Sometimes Mon Low can be treated similarly if Tue doesn't take it out
            profile_name = "Monday Low (Variant)"
            description = "Low formed on Monday. Often leads to a steady expansion."
        elif low_day == 2: # Wednesday
            profile_name = "Wednesday Low / Reversal"
            description = "Market manipulated Mon-Tue, Low formed Wednesday."
        elif low_day == 3: # Thursday
            profile_name = "Consolidation Thursday Reversal"
            description = "Consolidation Mon-Wed, Stop hunt Low on Thursday, then Reversal."
        elif low_day == 4: # Friday
            profile_name = "Seek & Destroy / Friday Low"
            description = "Choppy week, Low formed late on Friday."
            
    else: 
        # BEARISH PROFILES
        if high_day == 1: # Tuesday
            profile_name = "Classic Tuesday High"
            description = "High formed on Tuesday, followed by expansion lower."
        elif high_day == 0: # Monday
            profile_name = "Monday High (Variant)"
            description = "High formed on Monday. Often leads to a steady decline."
        elif high_day == 2: # Wednesday
            profile_name = "Wednesday High / Reversal"
            description = "Market manipulated Mon-Tue, High formed Wednesday."
        elif high_day == 3: # Thursday
            profile_name = "Consolidation Thursday Reversal"
            description = "Consolidation Mon-Wed, Stop hunt High on Thursday, then Reversal."
        elif high_day == 4: # Friday
            profile_name = "Seek & Destroy / Friday High"
            description = "Choppy week, High formed late on Friday."

    return {
        "Trend": trend,
        "Profile": profile_name,
        "Description": description,
        "Weekly_High": high_price,
        "Weekly_Low": low_price,
        "High_Day": high_date.strftime("%A"),
        "Low_Day": low_date.strftime("%A")
    }

# --- MAIN APP LOGIC ---

st.title(f"📊 ICT Weekly Profile Identifier: {selected_asset_name}")
st.markdown(f"Identifying weekly templates based on **{ticker_symbol}** price action.")

# 1. Load Data
with st.spinner('Fetching market data...'):
    df = get_data(ticker_symbol, weeks=lookback_weeks + 2) # Buffer

if df is not None:
    # Group data by Week
    weeks = df.groupby('Week_Start')
    week_keys = list(weeks.groups.keys())
    week_keys.sort(reverse=True) # Newest first

    # Select Week to Visualize
    # Filter out empty keys if any
    week_keys = [k for k in week_keys if not pd.isna(k)]
    
    if not week_keys:
        st.error("No valid weekly data found.")
    else:
        selected_week_start = st.selectbox(
            "Select Week to View Chart", 
            week_keys[:lookback_weeks],
            format_func=lambda x: f"Week of {x.strftime('%Y-%m-%d')}"
        )
        
        # Get Data for selected week
        current_week_df = weeks.get_group(selected_week_start).copy()
        
        # Analyze Profile
        analysis = identify_profile(current_week_df)
        
        # --- DISPLAY METRICS ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Weekly Trend", analysis["Trend"], delta_color="normal" if analysis["Trend"]=="Bullish" else "inverse")
        col2.metric("Detected Profile", analysis["Profile"])
        col3.metric("High Formed On", analysis["High_Day"])
        col4.metric("Low Formed On", analysis["Low_Day"])
        
        st.info(f"**Logic:** {analysis['Description']}")

        # --- PLOT CHART ---
        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=current_week_df['Date'],
            open=current_week_df['Open'],
            high=current_week_df['High'],
            low=current_week_df['Low'],
            close=current_week_df['Close'],
            name='Price'
        ))

        # Markers for High and Low
        # High Marker
        max_high_idx = current_week_df['High'].idxmax()
        fig.add_annotation(
            x=current_week_df.loc[max_high_idx, 'Date'],
            y=current_week_df.loc[max_high_idx, 'High'],
            text="Weekly High",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
        
        # Low Marker
        min_low_idx = current_week_df['Low'].idxmin()
        fig.add_annotation(
            x=current_week_df.loc[min_low_idx, 'Date'],
            y=current_week_df.loc[min_low_idx, 'Low'],
            text="Weekly Low",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=40
        )

        fig.update_layout(
            title=f"{selected_asset_name} - Week of {selected_week_start.strftime('%Y-%m-%d')}",
            xaxis_title="Date / Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=600,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- HISTORICAL TABLE ---
        st.subheader("Recent Weekly Profiles History")
        
        history_data = []
        for w_start in week_keys[:lookback_weeks]:
            w_df = weeks.get_group(w_start)
            res = identify_profile(w_df)
            res['Week Start'] = w_start.strftime('%Y-%m-%d')
            history_data.append(res)
        
        history_df = pd.DataFrame(history_data)
        # Reorder columns
        if not history_df.empty:
            history_df = history_df[['Week Start', 'Trend', 'Profile', 'High_Day', 'Low_Day']]
            st.dataframe(history_df, use_container_width=True)

else:
    st.error("Could not fetch data. Please check your connection or try again later.")

# --- FOOTER ---
st.markdown("---")
st.caption("Based on ICT Weekly Range Profile concepts. Profiles are estimated based on the Day of Week the High/Low was formed.")
