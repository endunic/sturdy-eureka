import sys
import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import requests
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

# ========================= CONFIGURATION =========================
forex_pairs = [
    'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X', 'USDCAD=X', 'USDCHF=X', 'USDJPY=X',
    'EURGBP=X', 'EURJPY=X', 'EURCHF=X', 'EURAUD=X', 'EURCAD=X', 'EURNZD=X',
    'GBPJPY=X', 'GBPCHF=X', 'GBPAUD=X', 'GBPCAD=X', 'GBPNZD=X',
    'AUDJPY=X', 'AUDCAD=X', 'AUDCHF=X', 'AUDNZD=X',
    'NZDJPY=X', 'NZDCAD=X', 'NZDCHF=X', 'CADJPY=X', 'CADCHF=X', 'CHFJPY=X'
]

# Dynamic start date: 729 days ago (safe limit for yfinance intraday data)
start_date = (datetime.now() - timedelta(days=729)).strftime('%Y-%m-%d')
SIGNAL_HISTORY_FILE = 'dashboard_output/signal_history.csv'
SIGNAL_HISTORY_COLUMNS = ['Timestamp', 'Pair', 'Signal', 'Z-Score', 'Duration (Periods)']

os.makedirs('dashboard_output', exist_ok=True)
IS_STREAMLIT = "streamlit" in sys.modules or "streamlit.runtime" in sys.modules

# --- DEFAULT CONFIG (Now editable in UI) ---
# These are used as initial values for Streamlit widgets
DEFAULT_INTERVAL = '1d'
DEFAULT_CORR = 0.5
DEFAULT_PVAL = 0.05
DEFAULT_WINDOW = 30
DEFAULT_Z_ENTRY = 2.0
DEFAULT_Z_STOP = 4.0
DEFAULT_Z_EXIT = 0.2
DEFAULT_COST = 0.15 # Transaction cost in Z-units per trade
DEFAULT_ACCOUNT = 100000 # Account size for risk calculation
DEFAULT_RISK = 0.01 # Risk per trade as a percentage of account

# ========================= CORE FUNCTIONS =========================
@st.cache_data(ttl=900)
def download_data(interval):
    """Downloads data with a persistent local file cache to prevent rate limits."""
    local_file = f'dashboard_output/close_prices_{interval}.csv'
    cache_expiry = timedelta(minutes=15)

    # 1. Check if local cache exists and is fresh
    if os.path.exists(local_file):
        mtime = datetime.fromtimestamp(os.path.getmtime(local_file))
        if datetime.now() - mtime < cache_expiry:
            try:
                df = pd.read_csv(local_file, index_col=0, parse_dates=True)
                if not df.empty:
                    return df
            except Exception:
                pass # Fallback to download if file is corrupt

    # 2. If no fresh cache, attempt download
    try:
        # Using yahooquery instead of yfinance for better API stability
        ticker = Ticker(forex_pairs, asynchronous=True)
        
        # yahooquery uses '1d', '1h', etc. but its interval logic is slightly different
        # history() returns a long-format DataFrame with MultiIndex [symbol, date]
        raw_data = ticker.history(start=start_date, interval=interval)
        
        if raw_data.empty:
            raise ValueError("Yahoo Finance (via yahooquery) returned empty data.")

        # Pivot the long-format data to match the expected dashboard format
        # columns = symbols, index = dates
        if isinstance(raw_data.index, pd.MultiIndex):
            processed_close = raw_data.reset_index().pivot(
                index='date', 
                columns='symbol', 
                values='adjclose'
            )
        else:
            # Fallback if only one ticker is returned
            processed_close = raw_data[['adjclose']]

        processed_close = processed_close.ffill().dropna(how='all')
        
        # Ensure column names match yfinance style (e.g., 'EURUSD=X')
        processed_close.columns = [str(c).upper() for c in processed_close.columns]
        
        processed_close.to_csv(local_file)
        return processed_close

    except Exception as e:
        # 3. If download fails (e.g., rate limit), try to use the stale cache as a fallback
        if os.path.exists(local_file):
            return pd.read_csv(local_file, index_col=0, parse_dates=True)
        raise ValueError(f"Failed to download data and no local cache available: {e}")


@st.cache_data(ttl=900)
def scan_cointegration(close_prices, corr_threshold, p_value_threshold):
    log_prices = np.log(close_prices).dropna(how='any')
    returns = log_prices.pct_change().dropna()
    corr_matrix = returns.corr()
    results = []

    def check_pair(p1, p2):
        corr = float(corr_matrix.loc[p1, p2])
        if abs(corr) > corr_threshold:
            try:
                X = sm.add_constant(log_prices[p2])
                model = sm.OLS(log_prices[p1], X).fit()
                hedge = float(model.params.iloc[1])
                spread = log_prices[p1] - hedge * log_prices[p2]
                pval = float(adfuller(spread, autolag='AIC')[1])
                if pval < p_value_threshold:
                    side2 = "Sell" if hedge > 0 else "Buy"
                    opp = "Buy" if hedge > 0 else "Sell"
                    return {
                        'Pair1': p1, 'Pair2': p2, 'Hedge_Ratio': round(hedge, 4),
                        'p_value': round(pval, 5), 'Correlation': round(corr, 3),
                        'Long_Action': f"Buy {p1} | {side2} {abs(hedge):.4f} {p2}",
                        'Short_Action': f"Sell {p1} | {opp} {abs(hedge):.4f} {p2}"
                    }
            except:
                return None
        return None

    pairs_to_check = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            pairs_to_check.append((corr_matrix.columns[i], corr_matrix.columns[j]))

    with ThreadPoolExecutor() as executor:
        tasks = [executor.submit(check_pair, p[0], p[1]) for p in pairs_to_check]
        for task in tasks:
            res = task.result()
            if res:
                results.append(res)

    df = pd.DataFrame(results)
    return df.sort_values('p_value').reset_index(drop=True) if not df.empty else df


def calculate_signals(pair1, pair2, hedge_ratio, close_prices, z_window, z_entry, z_exit, z_stop, interval):
    log_p = np.log(close_prices[[pair1, pair2]]).dropna()
    spread = log_p[pair1] - hedge_ratio * log_p[pair2]

    # Half-life calculation
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    # Ensure spread_lag and spread_diff are aligned
    common_index = spread_lag.index.intersection(spread_diff.index)
    if len(common_index) < 2: # Need at least 2 points for regression
        half_life = 999
    else:
        X_hl = sm.add_constant(spread_lag.loc[common_index])
        model_hl = sm.OLS(spread_diff.loc[common_index], X_hl).fit()
        lambda_val = model_hl.params.iloc[1]
        half_life = round(-np.log(2) / lambda_val, 1) if lambda_val < 0 else 999

    # Adjust half-life if using intraday intervals
    if interval == '4h': half_life = round(half_life / 6, 1)
    if interval == '1h': half_life = round(half_life / 24, 1)

    mean = spread.rolling(z_window).mean()
    std = spread.rolling(z_window).std()
    z_score = (spread - mean) / std
    
    # Ensure z_score has enough data points
    if z_score.empty:
        return {
            'spread': spread, 'z_score': pd.Series(), 'half_life': half_life,
            'latest_z': np.nan, 'signal': "N/A",
            'target_price1': np.nan, 'current_price1': np.nan, 'current_price2': np.nan,
            'regime': "N/A", 'spread_std': np.nan
        }

    latest_z = z_score.iloc[-1]

    # Advanced Regime Detection & Stationarity Verification
    recent_window = z_window * 2
    if len(spread) >= recent_window:
        recent_pval = float(adfuller(spread.tail(recent_window), autolag='AIC')[1])
        stationarity = "STABLE" if recent_pval < 0.05 else "DRIFTING/BROKEN"
    else:
        recent_pval = np.nan
        stationarity = "INSUFFICIENT DATA"

    regime = "STABLE" if len(mean) > 10 and abs((mean.iloc[-1] - mean.iloc[-10]) / 10) < 0.0005 else "DRIFTING"
    
    if latest_z >= z_stop or latest_z <= -z_stop:
        signal = "🛑 STOP LOSS"
    elif latest_z > z_entry:
        signal = "🔴 SHORT SPREAD"
    elif latest_z < -z_entry:
        signal = "🟢 LONG SPREAD"
    elif abs(latest_z) <= z_exit:
        signal = "✅ TAKE PROFIT"
    else:
        signal = "⚪ FLAT"

    # Ensure current_mean and current_std are not NaN
    current_mean = mean.iloc[-1] if not mean.empty else np.nan
    current_std = std.iloc[-1] if not std.empty else np.nan

    # Calculate Target Price for Pair1 (assuming Pair2 stays constant)
    p2_latest = close_prices[pair2].iloc[-1]
    if not np.isnan(current_mean) and not np.isnan(p2_latest):
        target_ln_p1 = current_mean + hedge_ratio * np.log(p2_latest)
        target_price1 = np.exp(target_ln_p1)
    else:
        target_price1 = np.nan

    return {
        'spread': spread, 'z_score': z_score, 'half_life': half_life,
        'latest_z': round(latest_z, 3), 'signal': signal,
        'target_price1': round(target_price1, 5) if not np.isnan(target_price1) else np.nan,
        'recent_pval': round(recent_pval, 4), 'stationarity': stationarity,
        'current_price1': round(close_prices[pair1].iloc[-1], 5),
        'current_price2': round(close_prices[pair2].iloc[-1], 5),
        'regime': regime, 'spread_std': round(current_std, 5) if not np.isnan(current_std) else np.nan
    }


def backtest_pair(z_score, pair_name, z_entry, z_exit, z_stop, transaction_cost_z):
    """Improved backtest with more stats and Kelly Criterion."""
    trades = []
    equity_curve = [0]
    in_position = 0  # 1 for Long, -1 for Short, 0 for Flat
    entry_z = 0.0
    
    for i in range(len(z_score)):
        z = z_score.iloc[i]
        
        if in_position == 0:
            if z < -z_entry:
                in_position = 1 # Enter Long
                entry_z = z
            elif z > z_entry:
                in_position = -1 # Enter Short
                entry_z = z
        
        elif in_position == 1: # In Long
            if z >= -z_exit or z <= -z_stop:
                pnl = (z - entry_z) - transaction_cost_z
                trades.append(pnl)
                equity_curve.append(equity_curve[-1] + pnl)
                in_position = 0
                
        elif in_position == -1: # In Short
            if z <= z_exit or z >= z_stop:
                pnl = (entry_z - z) - transaction_cost_z
                trades.append(pnl)
                equity_curve.append(equity_curve[-1] + pnl)
                in_position = 0
                
    if not trades:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, None # num_trades, win_rate, total_z_pnl, max_dd, expectancy, kelly, fig

    num_trades = len(trades)
    win_rate = len([t for t in trades if t > 0]) / num_trades * 100
    total_z_pnl = sum(trades)
    expectancy = total_z_pnl / num_trades

    # Kelly Criterion: (WinProb - (LossProb / WinLossRatio))
    win_prob = win_rate / 100
    avg_win = np.mean([t for t in trades if t > 0]) if any(t > 0 for t in trades) else 0
    avg_loss = abs(np.mean([t for t in trades if t < 0])) if any(t < 0 for t in trades) else 1
    win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    kelly = max(0, win_prob - ((1 - win_prob) / win_loss_ratio)) if win_loss_ratio != 0 else 0

    # Calculate Max Drawdown
    cum_pnl_series = pd.Series(equity_curve)
    running_max = cum_pnl_series.cummax()
    drawdown = running_max - cum_pnl_series
    max_dd = round(drawdown.max(), 2) if not drawdown.empty else 0.0

    # Plot Equity Curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_curve, marker='o', linestyle='-', color='blue')
    ax.set_title(f"Backtest Equity Curve: {pair_name}")
    ax.set_ylabel("Cumulative Z-PnL")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', lw=1)
    
    return num_trades, round(win_rate, 1), round(total_z_pnl, 2), max_dd, round(expectancy, 2), round(kelly, 4), fig


def plot_pair(pair1, pair2, hedge, spread, z_score, z_entry, z_exit, z_stop):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(spread, label='Spread')
    ax1.axhline(spread.mean(), color='red', linestyle='--', label='Mean')
    ax1.set_title(f'Spread: {pair1} vs {pair2} (Hedge: {hedge:.4f})')
    ax1.legend()
    
    ax2.plot(z_score, label='Z-Score', color='purple')
    ax2.axhline(z_entry, color='orange', linestyle='--', label=f'Entry ({z_entry})')
    ax2.axhline(-z_entry, color='orange', linestyle='--')
    ax2.axhline(z_exit, color='green', linestyle=':', label=f'Exit ({z_exit})')
    ax2.axhline(-z_exit, color='green', linestyle=':')
    ax2.axhline(z_stop, color='red', linestyle='-.', label=f'Stop ({z_stop})')
    ax2.axhline(-z_stop, color='red', linestyle='-.')
    ax2.axhline(0, color='black', alpha=0.5)
    ax2.set_title('Z-Score')
    ax2.legend()
    
    plt.tight_layout()
    return fig


def calculate_portfolio_exposure(active_list):
    exposure = {}
    for item in active_list:
        p1, p2 = item['Pair'].split('/')
        c1, c2 = p1[:3], p2[:3]
        mult = 1 if "LONG" in item.get('Signal', '') else -1
        exposure[c1] = exposure.get(c1, 0) + mult
        exposure[c2] = exposure.get(c2, 0) - mult
    return {k: v for k, v in exposure.items() if v != 0}


def send_telegram(message, token, chat_id):
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Telegram alert: {e}")
        return False


def load_signal_history():
    if not os.path.exists(SIGNAL_HISTORY_FILE):
        return pd.DataFrame(columns=SIGNAL_HISTORY_COLUMNS)
    try:
        df = pd.read_csv(SIGNAL_HISTORY_FILE, parse_dates=['Timestamp'])
        for col in SIGNAL_HISTORY_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        return df[SIGNAL_HISTORY_COLUMNS]
    except Exception as e:
        print(f"Error loading signal history: {e}. Returning empty DataFrame.")
        return pd.DataFrame(columns=SIGNAL_HISTORY_COLUMNS)


def track_signal(pair_name, signal_type, z_score, history_df):
    """Improved signal tracking: logs only if signal changes or if pair is new."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Check if the pair exists in the history and if the last signal is different
    pair_history = history_df[history_df['Pair'] == pair_name]
    
    should_log = False
    if pair_history.empty:
        should_log = True
    else:
        last_signal = pair_history.iloc[-1]['Signal']
        if last_signal != signal_type:
            should_log = True

    if should_log:
        new_entry = pd.DataFrame([{
            'Timestamp': now,
            'Pair': pair_name,
            'Signal': signal_type,
            'Z-Score': round(z_score, 3),
            'Duration (Periods)': None # This could be calculated and filled later if needed
        }])
        header = not os.path.exists(SIGNAL_HISTORY_FILE)
        new_entry.to_csv(SIGNAL_HISTORY_FILE, mode='a', header=header, index=False)
        return True
    return False


def calculate_signal_age(pair_name, current_signal, history_df, interval):
    """Calculates how many periods the current signal has been active."""
    pair_history = history_df[history_df['Pair'] == pair_name].sort_values('Timestamp', ascending=False)
    if pair_history.empty:
        return 1.0
    
    streak_start = pd.to_datetime(pair_history.iloc[0]['Timestamp'])
    for _, row in pair_history.iterrows():
        if row['Signal'] == current_signal:
            streak_start = pd.to_datetime(row['Timestamp'])
        else:
            break
    
    time_diff = (datetime.now() - streak_start).total_seconds()
    period_sec = 86400 if interval == '1d' else (14400 if interval == '4h' else 3600)
    return max(1.0, round(time_diff / period_sec, 1))


def run_terminal_scan():
    print(f"\n{'='*60}")
    print("FOREX COINTEGRATION SCANNER - TERMINAL MODE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    try:
        # Use default values for terminal mode
        close_prices = download_data(DEFAULT_INTERVAL)
        coint_pairs = scan_cointegration(close_prices, DEFAULT_CORR, DEFAULT_PVAL)
        if coint_pairs.empty:
            print("No cointegrated pairs found.")
            return

        history = load_signal_history()
        active = []
        for _, row in coint_pairs.head(15).iterrows(): # Scan top 15 pairs
            sig = calculate_signals(row['Pair1'], row['Pair2'], row['Hedge_Ratio'],
                                    close_prices, DEFAULT_WINDOW, DEFAULT_Z_ENTRY, DEFAULT_Z_EXIT, DEFAULT_Z_STOP, DEFAULT_INTERVAL)
            pair_name = f"{row['Pair1']}/{row['Pair2']}"
            track_signal(pair_name, sig['signal'], sig['latest_z'], history)
            if "FLAT" not in sig['signal']:
                active.append(f"{pair_name}: {sig['signal']} (Z={sig['latest_z']})")

        if active:
            print("ACTIVE SIGNALS:")
            for a in active:
                print(f"   • {a}")
        else:
            print("No active signals at the moment.")
    except Exception as e:
        print(f"Error during terminal scan: {e}")


# ========================= STREAMLIT UI =========================
def run_streamlit():
    st.set_page_config(page_title="Forex Cointegration Dashboard", layout="wide")
    st.title("📊 Forex Cointegration Trading Dashboard")
    st.markdown("**Statistical Arbitrage • Pairs Trading Scanner**")

    with st.sidebar:
        st.header("⚙️ Strategy Settings")
        interval = st.selectbox("Interval", ['1d', '1h'], index=0, help="Data aggregation interval.")
        corr_threshold = st.slider("Min Correlation", 0.3, 0.9, DEFAULT_CORR, step=0.05, help="Minimum absolute correlation between returns for pairs to be considered.")
        p_value_threshold = st.slider("Max p-value", 0.001, 0.1, DEFAULT_PVAL, step=0.001, help="Maximum p-value for ADF test on spread to confirm cointegration.")
        z_window = st.slider("Z-Score Window", 20, 60, DEFAULT_WINDOW, help="Rolling window for Z-score calculation.")
        z_entry = st.slider("Z Entry", 1.5, 3.0, DEFAULT_Z_ENTRY, step=0.1, help="Z-score threshold to enter a trade.")
        z_exit = st.slider("Z Exit", 0.1, 1.0, DEFAULT_Z_EXIT, step=0.1, help="Z-score threshold to exit a trade (mean reversion).")
        z_stop = st.slider("Z Stop", 3.0, 6.0, DEFAULT_Z_STOP, step=0.1, help="Z-score threshold for stop-loss (spread divergence).")
        transaction_cost_z = st.slider("Transaction Cost (Z-units)", 0.0, 0.5, DEFAULT_COST, step=0.01, help="Simulated transaction cost per trade in Z-score units.")

        st.markdown("---")
        st.header("💰 Risk Management")
        account_size = st.number_input("Account Size (USD)", min_value=1000, value=DEFAULT_ACCOUNT, step=1000, help="Your total trading account size in USD.")
        risk_per_trade = st.slider("Risk per Trade (%)", 0.1, 5.0, DEFAULT_RISK * 100, step=0.1, format="%.1f%%", help="Percentage of account to risk per trade.") / 100

        st.markdown("---")
        st.subheader("Telegram Alerts")
        use_tg = st.checkbox("Enable Telegram Alerts", value=False, help="Check to enable Telegram notifications for new signals.")
        tg_token = st.text_input("Telegram Bot Token", type="password", value="", help="Your Telegram bot token from BotFather.")
        tg_chat = st.text_input("Telegram Chat ID", type="password", value="", help="Your Telegram chat ID (get from @userinfobot).")

    # Initialize session state to persist results across reruns
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    if st.button("🔄 Run Full Analysis", type="primary", use_container_width=True):
        with st.spinner("Downloading data and scanning for cointegrated pairs..."):
            try:
                close_prices = download_data(interval)
                coint_pairs = scan_cointegration(close_prices, corr_threshold, p_value_threshold)
                if coint_pairs.empty:
                    st.error("No cointegrated pairs found with the current settings.")
                    st.session_state.analysis_results = None
                    return

                active_list = []
                watchlist = []
                history_df = load_signal_history() # Load history once for this run
                
                # Process top 20 cointegrated pairs
                for _, row in coint_pairs.head(20).iterrows():
                    sig = calculate_signals(row['Pair1'], row['Pair2'], row['Hedge_Ratio'],
                                            close_prices, z_window, z_entry, z_exit, z_stop, interval)
                    pair_name = f"{row['Pair1']}/{row['Pair2']}"
                    
                    # Track signal changes
                    track_signal(pair_name, sig['signal'], sig['latest_z'], history_df)
                    
                    # Calculate Signal Age
                    age = calculate_signal_age(pair_name, sig['signal'], history_df, interval)

                    if "FLAT" not in sig['signal']:
                        active_list.append({
                            'Pair': pair_name,
                            'Z-Score': sig['latest_z'],
                            'Signal': sig['signal'],
                            'Hedge': row['Hedge_Ratio'],
                            'Regime': sig['regime'],
                            'Age (Bars)': age
                        })
                    elif abs(sig['latest_z']) > 1.5: # Watchlist for Z-scores between 1.5 and entry
                        watchlist.append({
                            'Pair': pair_name,
                            'Z-Score': sig['latest_z'],
                            'Status': "👀 Approaching",
                            'Regime': sig['regime'],
                            'Age (Bars)': age
                        })

                # Trigger Telegram Alert only on the initial button click
                if use_tg and tg_token and tg_chat and active_list:
                    msg = f"🚨 *Active Signals Detected* ({len(active_list)})\n\n" + \
                          "\n".join([f"{a['Pair']}: {a['Signal']} (Z={a['Z-Score']})" for a in active_list[:5]])
                    if send_telegram(msg, tg_token, tg_chat):
                        st.success("✅ Telegram alert sent!")
                    else:
                        st.warning("Failed to send Telegram alert. Check token/chat ID.")

                # Store results in session state
                st.session_state.analysis_results = {
                    'close_prices': close_prices,
                    'coint_pairs': coint_pairs,
                    'active_list': active_list,
                    'watchlist': watchlist,
                    'settings': { # Store settings used for analysis
                        'interval': interval, 'z_window': z_window,
                        'z_entry': z_entry, 'z_exit': z_exit, 'z_stop': z_stop,
                        'account_size': account_size, 'risk_per_trade': risk_per_trade,
                        'transaction_cost_z': transaction_cost_z
                    }
                }
                st.success(f"✅ Analysis complete! Found **{len(coint_pairs)}** cointegrated pairs.")
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.session_state.analysis_results = None

    # Render analysis UI if results exist in session state
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        close_prices = res['close_prices']
        coint_pairs = res['coint_pairs']
        active_list = res['active_list']
        watchlist = res['watchlist']
        settings = res['settings'] # Retrieve settings from session state

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🚨 Active Signals", "📜 Signal History", "📋 All Pairs",
            "🔍 Pair Deep Dive", "📈 Backtest & Performance" # Renamed tab
        ])

        # Tab 1: Active Signals
        with tab1:
            st.subheader("Active Trading Signals")
            if active_list:
                st.dataframe(pd.DataFrame(active_list), use_container_width=True)
                exposure = calculate_portfolio_exposure(active_list)
                if exposure:
                    st.subheader("Portfolio Currency Exposure")
                    st.write(exposure)
                    if any(abs(v) > 2 for v in exposure.values()):
                        st.warning("⚠️ High currency concentration detected! Consider diversifying.")
                else:
                    st.info("No significant currency exposure from active signals.")
            else:
                st.info("No active signals at the moment.")
            
            if watchlist:
                st.subheader("Watchlist (Approaching Entry)")
                st.dataframe(pd.DataFrame(watchlist), use_container_width=True)

        # Tab 2: Signal History
        with tab2:
            st.subheader("Signal History")
            if os.path.exists(SIGNAL_HISTORY_FILE):
                history = load_signal_history() # Reload to get latest updates
                st.dataframe(history.sort_values('Timestamp', ascending=False), use_container_width=True)
                st.download_button(
                    "📥 Download Signal History CSV",
                    history.to_csv(index=False).encode('utf-8'),
                    "signal_history.csv",
                    "text/csv",
                    key='download_history_csv'
                )
            else:
                st.info("No signal history yet.")

        # Tab 3: All Pairs
        with tab3:
            st.subheader("Top Cointegrated Pairs (by p-value)")
            st.dataframe(coint_pairs.head(25), use_container_width=True)
            st.download_button(
                "📥 Download All Cointegrated Pairs CSV",
                coint_pairs.to_csv(index=False).encode('utf-8'),
                "cointegrated_pairs.csv",
                "text/csv",
                key='download_coint_pairs_csv'
            )

        # Tab 4: Pair Deep Dive
        with tab4:
            st.subheader("🔍 Pair Deep Dive Analysis")
            
            # Use the settings from when the analysis was run
            current_z_entry = settings['z_entry']
            current_z_stop = settings['z_stop']
            current_account_size = settings['account_size']
            current_risk_per_trade = settings['risk_per_trade']

            idx = st.selectbox("Select Pair", range(len(coint_pairs)),
                               format_func=lambda x: f"{coint_pairs.iloc[x]['Pair1']}/{coint_pairs.iloc[x]['Pair2']}",
                               key='deep_dive_select')
            row = coint_pairs.iloc[idx]
            sig = calculate_signals(row['Pair1'], row['Pair2'], row['Hedge_Ratio'],
                                    close_prices, settings['z_window'], current_z_entry, settings['z_exit'], current_z_stop, settings['interval'])
            
            st.markdown("---")
            st.subheader(f"Details for {row['Pair1']}/{row['Pair2']}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Latest Z-Score", f"{sig['latest_z']:.3f}", delta_color="off")
            col2.metric("Signal", sig['signal'])
            col3.metric("Hedge Ratio", f"{row['Hedge_Ratio']:.4f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("Half-Life (Periods)", f"{sig['half_life']:.1f}")
            col5.metric("Recent Stationarity", sig['stationarity'], help="ADF test on recent window.")
            col6.metric("p-value", f"{row['p_value']:.5f}")

            st.markdown("---")
            st.subheader("Risk Sizing & Quality")
            
            # Quality Ranking
            quality = "⭐ LOW (High risk of breakdown)"
            if row['p_value'] < 0.01 and sig['half_life'] < 15 and sig['stationarity'] == "STABLE":
                quality = "⭐⭐⭐ HIGH (Strong Statistical Reversion)"
            elif row['p_value'] < 0.03:
                quality = "⭐⭐ MEDIUM"
            st.metric("Signal Quality", quality)

            # Risk Sizing Calculator
            risk_amount_usd = current_account_size * current_risk_per_trade
            
            # Ensure spread_std is not NaN or zero before division
            if not np.isnan(sig['spread_std']) and sig['spread_std'] > 0:
                z_risk_dist = current_z_stop - current_z_entry
                if z_risk_dist > 0:
                    # Notional = Risk_USD / (Risk_in_Z * Std_Dev_of_Spread)
                    # This is the total notional for the spread.
                    # For a pair trade, this is often split between the two legs.
                    total_notional_spread = risk_amount_usd / (z_risk_dist * sig['spread_std'])
                    notional_per_leg = total_notional_spread / 2 # A common simplification
                else:
                    total_notional_spread = 0
                    notional_per_leg = 0
            else:
                total_notional_spread = 0
                notional_per_leg = 0

            # Lot Size Calculation (Standard Lot = 100,000 units)
            units_p1 = notional_per_leg / sig['current_price1']
            lots_p1 = units_p1 / 100000

            st.markdown(f"**Risk Amount per Trade:** ${risk_amount_usd:,.2f} ({current_risk_per_trade*100:.1f}% of ${current_account_size:,.0f})")
            st.markdown(f"**Volatility-Adjusted Notional (Total Spread):** ${total_notional_spread:,.2f}")
            st.markdown(f"**Suggested Notional per Leg:** ${notional_per_leg:,.2f} (~{units_p1:,.0f} units)")
            st.info(f"💡 **Trade Execution:** Open approximately **{lots_p1:.2f} Standard Lots** for {row['Pair1']}.")
            st.caption(f"*(Based on Z-Entry {current_z_entry}, Z-Stop {current_z_stop}, and Spread Std Dev {sig['spread_std']:.5f})*")

            st.markdown("---")
            st.subheader("Price Targets")
            st.metric(f"Current {row['Pair1']} Price", f"{sig['current_price1']:.5f}")
            st.metric(f"Target {row['Pair1']} Price (at mean)", f"{sig['target_price1']:.5f}")
            st.caption(f"*(Assuming {row['Pair2']} price remains constant at {sig['current_price2']:.5f})*")

            st.markdown("---")
            st.subheader("Z-Score Plot")
            st.pyplot(plot_pair(row['Pair1'], row['Pair2'], row['Hedge_Ratio'], 
                                sig['spread'], sig['z_score'], 
                                current_z_entry, settings['z_exit'], current_z_stop))

        # Tab 5: Backtest & Performance
        with tab5:
            st.subheader("📈 Backtest & Performance Summary")
            
            # Use the settings from when the analysis was run
            current_z_entry = settings['z_entry']
            current_z_exit = settings['z_exit']
            current_z_stop = settings['z_stop']
            current_transaction_cost_z = settings['transaction_cost_z']

            idx = st.selectbox("Select Pair for Backtest", range(len(coint_pairs)),
                               format_func=lambda x: f"{coint_pairs.iloc[x]['Pair1']}/{coint_pairs.iloc[x]['Pair2']}",
                               key="bt_select")
            row = coint_pairs.iloc[idx]
            sig = calculate_signals(row['Pair1'], row['Pair2'], row['Hedge_Ratio'],
                                    close_prices, settings['z_window'], current_z_entry, current_z_exit, current_z_stop, settings['interval'])
            
            # Run backtest with current settings
            num_trades, win_rate, pnl, max_dd, expectancy, kelly_val, fig = backtest_pair(
                sig['z_score'], f"{row['Pair1']}/{row['Pair2']}",
                current_z_entry, current_z_exit, current_z_stop, current_transaction_cost_z
            )
            
            st.markdown("---")
            st.subheader(f"Backtest Results for {row['Pair1']}/{row['Pair2']}")
            
            if num_trades > 0:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Trades", num_trades)
                col2.metric("Win Rate", f"{win_rate:.1f}%")
                col3.metric("Total PnL (Z-units)", f"{pnl:.2f}")

                col4, col5, col6 = st.columns(3)
                col4.metric("Max Drawdown (Z-units)", f"{max_dd:.2f}")
                col5.metric("Expectancy (Z-units/trade)", f"{expectancy:.2f}")
                col6.metric("Kelly Criterion (Fraction)", f"{kelly_val:.4f}")
                st.caption(f"*(Kelly Criterion suggests risking {kelly_val*100:.2f}% of capital per trade if conditions hold)*")

                st.pyplot(fig)
            else:
                st.info("No trades executed in the backtest with current parameters.")

    st.caption("💡 You can also run in terminal: `python forex_cointegration_dashboard.py`")


def main():
    if IS_STREAMLIT:
        run_streamlit()
    else:
        run_terminal_scan()


if __name__ == "__main__":
    main()
