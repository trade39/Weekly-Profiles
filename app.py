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
    Fetches daily data from yfinance with error handling.
    """
    # Fetch extra buffer to ensure we can calculate Previous Week High/Low
    period_days = weeks * 7 + 30 
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
        
        if data.empty:
            st.error(f"⚠️ Yahoo Finance returned no data for **{ticker}**.")
            st.warning("Tip: Try checking the 'Use ETF Tickers' box in the sidebar.")
            return None

        # Flatten MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Identify the start of the week (Monday)
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
    
    # Identify Day of Week for High and Low
    high_date = week_df.loc[week_df['High'].idxmax(), 'Date']
    low_date = week_df.loc[week_df['Low'].idxmin(), 'Date']
    
    if hasattr(high_date, 'item'): high_date = high_date.item()
    if hasattr(low_date, 'item'): low_date = low_date.item()
    
    high_day_num = high_date.weekday()
    low_day_num = low_date.weekday()
    
    profile_name = "Undefined"
    description = "Pattern does not fit strict classic definitions."

    if is_bullish:
        if low_day_num == 1: profile_name = "Classic Tuesday Low"
        elif low_day_num == 0: profile_name = "Monday Low"
        elif low_day_num == 2: profile_name = "Wednesday Low / Reversal"
        elif low_day_num == 3: profile_name = "Consolidation Thursday Reversal"
        elif low_day_num == 4: profile_name = "Seek & Destroy / Friday Low"
    else: 
        if high_day_num == 1: profile_name = "Classic Tuesday High"
        elif high_day_num == 0: profile_name = "Monday High"
        elif high_day_num == 2: profile_name = "Wednesday High / Reversal"
        elif high_day_num == 3: profile_name = "Consolidation Thursday Reversal"
        elif high_day_num == 4: profile_name = "Seek & Destroy / Friday High"

    # Assign description based on profile name logic (simplified for brevity)
    if description == "Pattern does not fit strict classic definitions.":
        if "Classic" in profile_name: description = "Standard expansion profile."
        elif "Reversal" in profile_name: description = "Mid-week manipulation and reversal."
        elif "Consolidation" in profile_name: description = "Range bound until late week."
        elif "Seek" in profile_name: description = "Volatile conditions, late week moves."
        elif "Monday" in profile_name: description = "Early week move that sustains."

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

# 1. Load Data
with st.spinner(f'Fetching data for {ticker_symbol}...'):
    fetch_weeks = max(lookback_weeks, stats_lookback) + 5 # Extra buffer for PWH/PWL
    df = get_data(ticker_symbol, weeks=fetch_weeks)

if df is not None:
    weeks = df.groupby('Week_Start')
    week_keys = list(weeks.groups.keys())
    week_keys.sort(reverse=True)
    week_keys = [k for k in week_keys if not pd.isna(k)]
    
    if not week_keys:
        st.error("No valid weekly data found.")
    else:
        # Pre-calculate stats
        stats_data = []
        for w_start in week_keys[:stats_lookback]:
            w_df = weeks.get_group(w_start)
            if len(w_df) >= 3:
                res = identify_profile(w_df)
                res['Week Start'] = w_start
                stats_data.append(res)
        stats_df = pd.DataFrame(stats_data)

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["🔎 Current Analysis", "📈 Statistical Probability", "🗓️ Seasonal Tendency"])

        # ==========================
        # TAB 1: CURRENT ANALYSIS (With PWH/PWL)
        # ==========================
        with tab1:
            selected_week_start = st.selectbox(
                "Select Week to View Chart", 
                week_keys[:lookback_weeks],
                format_func=lambda x: f"Week of {x.strftime('%Y-%m-%d')}"
            )
            
            current_week_df = weeks.get_group(selected_week_start).copy()
            analysis = identify_profile(current_week_df)
            
            # --- CALCULATE PWH / PWL ---
            pwh, pwl = None, None
            
            # Find index of selected week in the master list
            if selected_week_start in week_keys:
                idx = week_keys.index(selected_week_start)
                # Check if previous week exists (it's the next item in the list because it's sorted Reverse=True)
                if idx + 1 < len(week_keys):
                    prev_week_start = week_keys[idx + 1]
                    prev_week_df = weeks.get_group(prev_week_start)
                    pwh = prev_week_df['High'].max()
                    pwl = prev_week_df['Low'].min()

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Weekly Trend", analysis["Trend"], delta_color="normal" if analysis["Trend"]=="Bullish" else "inverse")
            c2.metric("Detected Profile", analysis["Profile"])
            c3.metric("High Formed On", analysis["High_Day"])
            c4.metric("Low Formed On", analysis["Low_Day"])
            
            st.info(f"**Logic:** {analysis['Description']}")

            # Chart
            fig = go.Figure()
            
            # Candlesticks
            fig.add_trace(go.Candlestick(
                x=current_week_df['Date'], open=current_week_df['Open'],
                high=current_week_df['High'], low=current_week_df['Low'],
                close=current_week_df['Close'], name='Price'
            ))

            # Add PWH / PWL Lines
            if pwh is not None:
                fig.add_hline(y=pwh, line_dash="dash", line_color="orange", annotation_text="PWH (Prev Week High)", annotation_position="top left")
            if pwl is not None:
                fig.add_hline(y=pwl, line_dash="dash", line_color="orange", annotation_text="PWL (Prev Week Low)", annotation_position="bottom left")

            # High/Low Markers for Current Week
            max_idx = current_week_df['High'].idxmax()
            min_idx = current_week_df['Low'].idxmin()
            
            fig.add_annotation(x=current_week_df.loc[max_idx, 'Date'], y=current_week_df.loc[max_idx, 'High'],
                               text="High", showarrow=True, arrowhead=1, ay=-40)
            fig.add_annotation(x=current_week_df.loc[min_idx, 'Date'], y=current_week_df.loc[min_idx, 'Low'],
                               text="Low", showarrow=True, arrowhead=1, ay=40)

            fig.update_layout(
                title=f"{selected_asset_name} - Week of {selected_week_start.strftime('%Y-%m-%d')}",
                xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False,
                height=600, template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            # History Table
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
                col_a, col_b = st.columns(2)
                
                with col_a:
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
                
                with col_b:
                    st.subheader("Profile Distribution")
                    profile_counts = stats_df['Profile'].value_counts().reset_index()
                    profile_counts.columns = ['Profile', 'Count']
                    fig_prof = px.pie(profile_counts, names='Profile', values='Count', hole=0.4)
                    fig_prof.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_prof, use_container_width=True)
            else:
                st.warning("Not enough data.")

        # ==========================
        # TAB 3: SEASONAL TENDENCY (Average Weekly Path)
        # ==========================
        with tab3:
            st.markdown("### Average Weekly Path (Composite Model)")
            st.caption(f"Showing the average price movement (%) by day of the week over the last {len(stats_df)} weeks.")
            
            # Logic: Collect % change of Close relative to Week Open for each day
            composite_data = {0:[], 1:[], 2:[], 3:[], 4:[]} # 0=Mon, 4=Fri
            
            for w_start in week_keys[:stats_lookback]:
                w_df = weeks.get_group(w_start)
                if len(w_df) > 1:
                    try:
                        week_open = w_df.iloc[0]['Open']
                        if hasattr(week_open, 'item'): week_open = week_open.item()
                        
                        # Calculate % change for each day in this week
                        for _, row in w_df.iterrows():
                            day_num = row['Date'].weekday()
                            if day_num <= 4: # Ignore weekends
                                close_val = row['Close']
                                if hasattr(close_val, 'item'): close_val = close_val.item()
                                
                                pct_change = ((close_val - week_open) / week_open) * 100
                                composite_data[day_num].append(pct_change)
                    except Exception:
                        continue

            # Calculate Averages
            days_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            avg_changes = []
            
            for i in range(5):
                vals = composite_data[i]
                if vals:
                    avg_changes.append(sum(vals) / len(vals))
                else:
                    avg_changes.append(0.0)

            # Plot Line Chart
            comp_df = pd.DataFrame({"Day": days_labels, "Avg % Change": avg_changes})
            
            fig_comp = px.line(comp_df, x="Day", y="Avg % Change", markers=True, 
                               title=f"Average Weekly Path - {selected_asset_name}")
            
            # Add a zero line
            fig_comp.add_hline(y=0, line_dash="dot", line_color="gray")
            
            # Color logic: Green line if Friday > Monday, else Red
            line_color = "#00CC96" if avg_changes[-1] > avg_changes[0] else "#EF553B"
            fig_comp.update_traces(line_color=line_color, line_width=3)
            
            fig_comp.update_layout(template="plotly_dark", yaxis_title="% Change from Monday Open")
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.info("""
            **How to read this chart:**
            - **0.00 Line:** Represents Monday's Opening Price.
            - **The Line:** Shows the average position of price relative to the opening, day by day.
            - **Use Case:** If the line bottoms out on Tuesday, it suggests the asset statistically creates the Low of the Week on Tuesday (Classic Tuesday Low).
            """)

else:
    st.info("Please select an asset.")

# --- FOOTER ---
# Market Scanner logic remains if you kept it, but excluded here to keep focus on the requested updates.
st.markdown("---")
st.caption("Based on ICT Weekly Range Profile concepts.")
