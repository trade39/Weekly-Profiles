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

# 1. Fallback Toggle
use_etf = st.sidebar.checkbox("Use ETF Tickers (More Stable)", value=False, 
    help="Check this if Futures data (ES=F, GC=F) fails to load. ETFs (SPY, GLD) are more reliable on the free API.")

# 2. Asset Selection
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

# 3. Lookback Periods
st.sidebar.subheader("Analysis Settings")
lookback_weeks = st.sidebar.slider("Recent Weeks to Display", min_value=1, max_value=20, value=4)
stats_lookback = st.sidebar.slider("Statistical Analysis Range (Weeks)", min_value=10, max_value=104, value=52, 
                                   help="How far back to look for probability stats.")

# --- HELPER FUNCTIONS ---

def get_data(ticker, weeks=52):
    """
    Fetches daily data from yfinance with error handling and debugging.
    """
    period_days = weeks * 7 + 21 # Buffer for PWH/PWL calculation
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)
    
    try:
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
        
        if data.empty:
            st.error(f"⚠️ Yahoo Finance returned no data for **{ticker}**.")
            st.warning("Tip: Try checking the 'Use ETF Tickers' box in the sidebar.")
            return None

        # --- FIX: Flatten MultiIndex Columns ---
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Reset index to make Date a column
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Identify the start of the week (Monday) for grouping
        data['Week_Start'] = data['Date'].apply(lambda x: x - timedelta(days=x.weekday()))
        
        return data

    except Exception as e:
        st.error(f"Critical Error fetching data for {ticker}: {e}")
        return None

def identify_profile(week_df):
    """
    Analyzes a single week's dataframe to identify the ICT Weekly Profile.
    """
    if week_df.empty:
        return {"Type": "Insufficient Data", "Desc": "No data available"}

    try:
        open_price = week_df.iloc[0]['Open'].item()
        close_price = week_df.iloc[-1]['Close'].item()
    except AttributeError:
        open_price = week_df.iloc[0]['Open']
        close_price = week_df.iloc[-1]['Close']

    high_price = week_df['High'].max()
    low_price = week_df['Low'].min()
    
    is_bullish = close_price > open_price
    trend = "Bullish" if is_bullish else "Bearish"
    
    high_date = week_df.loc[week_df['High'].idxmax(), 'Date']
    low_date = week_df.loc[week_df['Low'].idxmin(), 'Date']
    
    if hasattr(high_date, 'item'): high_date = high_date.item()
    if hasattr(low_date, 'item'): low_date = low_date.item()
    
    high_day_num = high_date.weekday()
    low_day_num = low_date.weekday()
    
    profile_name = "Undefined"
    description = "Pattern does not fit strict classic definitions."

    if is_bullish:
        if low_day_num == 1: profile_name, description = "Classic Tuesday Low", "Low formed on Tuesday, followed by expansion higher."
        elif low_day_num == 0: profile_name, description = "Monday Low", "Low formed on Monday. Often leads to a steady expansion."
        elif low_day_num == 2: profile_name, description = "Wednesday Low / Reversal", "Market manipulated Mon-Tue, Low formed Wednesday."
        elif low_day_num == 3: profile_name, description = "Consolidation Thursday Reversal", "Consolidation Mon-Wed, Stop hunt Low on Thursday, then Reversal."
        elif low_day_num == 4: profile_name, description = "Seek & Destroy / Friday Low", "Choppy week, Low formed late on Friday."
    else: 
        if high_day_num == 1: profile_name, description = "Classic Tuesday High", "High formed on Tuesday, followed by expansion lower."
        elif high_day_num == 0: profile_name, description = "Monday High", "High formed on Monday. Often leads to a steady decline."
        elif high_day_num == 2: profile_name, description = "Wednesday High / Reversal", "Market manipulated Mon-Tue, High formed Wednesday."
        elif high_day_num == 3: profile_name, description = "Consolidation Thursday Reversal", "Consolidation Mon-Wed, Stop hunt High on Thursday, then Reversal."
        elif high_day_num == 4: profile_name, description = "Seek & Destroy / Friday High", "Choppy week, High formed late on Friday."

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

def calculate_seasonal_path(weeks_dict, lookback):
    """
    Calculates the average percentage move for each day of the week
    relative to the Monday Open.
    """
    seasonal_data = []
    
    # Iterate through recent weeks
    week_keys = list(weeks_dict.groups.keys())
    week_keys.sort(reverse=True)
    
    for w_start in week_keys[:lookback]:
        w_df = weeks_dict.get_group(w_start).copy()
        if len(w_df) < 2: continue # Skip partial weeks
        
        # Get Monday Open Price
        mon_open = w_df.iloc[0]['Open']
        if hasattr(mon_open, 'item'): mon_open = mon_open.item()
        
        # Calculate % deviation for each day available
        for _, row in w_df.iterrows():
            day_num = row['Date'].weekday()
            close_price = row['Close']
            pct_change = ((close_price - mon_open) / mon_open) * 100
            
            seasonal_data.append({
                "DayNum": day_num,
                "PctChange": pct_change,
                "DayName": row['Date'].strftime("%A")
            })
            
    if not seasonal_data:
        return None
        
    df_seas = pd.DataFrame(seasonal_data)
    # Group by Day Number and get Mean
    avg_path = df_seas.groupby('DayNum').agg({'PctChange': 'mean', 'DayName': 'first'}).reset_index()
    return avg_path

# --- MAIN APP LOGIC ---

st.title(f"📊 ICT Weekly Profile Identifier: {selected_asset_name}")
st.markdown(f"Analysis based on **{ticker_symbol}** price action.")

# 1. Load Data
with st.spinner(f'Fetching data for {ticker_symbol}...'):
    fetch_weeks = max(lookback_weeks, stats_lookback) + 3 # Extra buffer for Prev Week
    df = get_data(ticker_symbol, weeks=fetch_weeks)

if df is not None:
    weeks = df.groupby('Week_Start')
    week_keys = list(weeks.groups.keys())
    week_keys.sort(reverse=True)
    week_keys = [k for k in week_keys if not pd.isna(k)]
    
    if not week_keys:
        st.error("No valid weekly data found.")
    else:
        # --- PROCESS STATS ---
        stats_data = []
        for w_start in week_keys[:stats_lookback]:
            w_df = weeks.get_group(w_start)
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
            # Dropdown for week selection
            selected_week_start = st.selectbox(
                "Select Week to View Chart", 
                week_keys[:lookback_weeks],
                format_func=lambda x: f"Week of {x.strftime('%Y-%m-%d')}"
            )
            
            # Identify Previous Week for PWH/PWL
            # Find index of selected week
            sel_idx = week_keys.index(selected_week_start)
            prev_week_data = None
            
            # If there is a week before this one in our data
            if sel_idx + 1 < len(week_keys):
                prev_week_start = week_keys[sel_idx + 1]
                prev_week_df = weeks.get_group(prev_week_start)
                prev_week_data = {
                    "PWH": prev_week_df['High'].max(),
                    "PWL": prev_week_df['Low'].min()
                }

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

            # --- LIQUIDITY LEVELS (PWH / PWL) ---
            if prev_week_data:
                # Add PWH Line
                fig.add_hline(y=prev_week_data['PWH'], line_dash="dash", line_color="orange", annotation_text="PWH", annotation_position="top right")
                # Add PWL Line
                fig.add_hline(y=prev_week_data['PWL'], line_dash="dash", line_color="orange", annotation_text="PWL", annotation_position="bottom right")

            # Annotations (Current High/Low)
            max_idx = current_week_df['High'].idxmax()
            min_idx = current_week_df['Low'].idxmin()
            
            fig.add_annotation(x=current_week_df.loc[max_idx, 'Date'], y=current_week_df.loc[max_idx, 'High'],
                               text="High", showarrow=True, arrowhead=1, ay=-40)
            fig.add_annotation(x=current_week_df.loc[min_idx, 'Date'], y=current_week_df.loc[min_idx, 'Low'],
                               text="Low", showarrow=True, arrowhead=1, ay=40)

            fig.update_layout(
                title=f"{selected_asset_name} - Week of {selected_week_start.strftime('%Y-%m-%d')}",
                xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False,
                height=600, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Recent History Table
            st.subheader("Recent History")
            display_cols = ['Week Start', 'Trend', 'Profile', 'High_Day', 'Low_Day']
            if not stats_df.empty:
                 st.dataframe(stats_df[display_cols].head(lookback_weeks), use_container_width=True)

        # ==========================
        # TAB 2: STATISTICAL PROBABILITY
        # ==========================
        with tab2:
            st.markdown(f"### Statistical Analysis (Last {len(stats_df)} Weeks)")
            
            if not stats_df.empty:
                
                # --- NEW: SEASONAL PATH CHART ---
                st.subheader("Average Weekly Path (Composite)")
                st.caption(f"Average price movement (Mon-Fri) over the last {stats_lookback} weeks. This reveals the asset's 'Personality'.")
                
                avg_path = calculate_seasonal_path(weeks, stats_lookback)
                
                if avg_path is not None:
                    fig_seas = px.line(avg_path, x="DayName", y="PctChange", markers=True, 
                                       title=f"Composite Weekly Path (% Change from Monday Open)",
                                       labels={"PctChange": "Average % Move", "DayName": "Day of Week"})
                    
                    # Add zero line
                    fig_seas.add_hline(y=0, line_dash="dot", line_color="white")
                    
                    fig_seas.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_seas, use_container_width=True)
                else:
                    st.warning("Not enough data to calculate seasonal path.")

                st.markdown("---")
                
                # 1. Day of Week Probability
                st.subheader("Day of Week Probability")
                days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                high_counts = stats_df['High_Day'].value_counts().reindex(days_order, fill_value=0)
                low_counts = stats_df['Low_Day'].value_counts().reindex(days_order, fill_value=0)
                
                counts_df = pd.DataFrame({"Day": days_order, "High Formation": high_counts.values, "Low Formation": low_counts.values})
                melted_counts = counts_df.melt(id_vars="Day", var_name="Type", value_name="Count")
                
                fig_days = px.bar(melted_counts, x="Day", y="Count", color="Type", barmode="group",
                    color_discrete_map={"High Formation": "#EF553B", "Low Formation": "#00CC96"})
                fig_days.update_layout(template="plotly_dark")
                st.plotly_chart(fig_days, use_container_width=True)
                
                # 2. Profile & Trend
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Profile Distribution")
                    profile_counts = stats_df['Profile'].value_counts().reset_index()
                    profile_counts.columns = ['Profile', 'Count']
                    fig_prof = px.pie(profile_counts, names='Profile', values='Count', hole=0.4)
                    fig_prof.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_prof, use_container_width=True)
                
                with col_b:
                    st.subheader("Trend Bias")
                    trend_counts = stats_df['Trend'].value_counts().reset_index()
                    trend_counts.columns = ['Trend', 'Count']
                    fig_trend = px.bar(trend_counts, x='Trend', y='Count', color='Trend',
                                       color_discrete_map={"Bullish": "#00CC96", "Bearish": "#EF553B"})
                    fig_trend.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_trend, use_container_width=True)

            else:
                st.warning("Not enough data to generate statistics.")

else:
    st.info("Please select an asset. If data fails to load, check the 'Use ETF Tickers' box.")

# --- SIDEBAR: MARKET SCREENER ---
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Market Scanner")

if st.sidebar.button("Scan All Assets"):
    st.markdown("### 🔍 Market Wide Analysis (Current Week)")
    scanner_results = []
    progress_bar = st.progress(0)
    total_assets = len(asset_map)
    
    for i, (name, ticker) in enumerate(asset_map.items()):
        progress_bar.progress((i + 1) / total_assets)
        try:
            d = get_data(ticker, weeks=4) 
            if d is not None:
                w_groups = d.groupby('Week_Start')
                keys = list(w_groups.groups.keys())
                keys.sort(reverse=True)
                if keys:
                    curr_df = w_groups.get_group(keys[0])
                    res = identify_profile(curr_df)
                    scanner_results.append({
                        "Asset": name,
                        "Trend": res['Trend'],
                        "Profile": res['Profile'],
                        "High Day": res['High_Day'],
                        "Low Day": res['Low_Day']
                    })
        except Exception: pass

    progress_bar.empty()
    if scanner_results:
        scan_df = pd.DataFrame(scanner_results)
        def highlight_trend(val):
            return f'color: {"#00CC96" if val == "Bullish" else "#EF553B"}; font-weight: bold'
        st.dataframe(scan_df.style.applymap(highlight_trend, subset=['Trend']), use_container_width=True)
    else:
        st.warning("No data found during scan.")
