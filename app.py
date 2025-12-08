import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import akshare as ak
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(
    page_title="BHMM A-Share Pro",
    page_icon="🇨🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore")

# 保持“彭博风”样式
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"] {
        background-color: rgba(28, 31, 46, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px; border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stMetricValue"] { font-family: 'Roboto Mono', monospace; color: #E0E0E0; }
    div.stButton > button {
        background: linear-gradient(90deg, #D32F2F 0%, #FF5252 100%);
        color: white; border: none; font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 数据引擎 (同步 Deepnote 逻辑)
# ==========================================

@st.cache_data(ttl=24*3600)
def get_all_a_share_list():
    """获取全市场列表"""
    try:
        df = ak.stock_zh_a_spot_em()
        df = df[['代码', '名称']]
        df['Display'] = df['代码'] + " | " + df['名称']
        return df, True
    except:
        return pd.DataFrame(), False

@st.cache_data(ttl=3600)
def format_ticker_for_yfinance(raw_code, raw_name="Unknown"):
    raw_code = str(raw_code).strip()
    if raw_code.startswith("6") or raw_code.startswith("9"): suffix = ".SS"
    elif raw_code.startswith("0") or raw_code.startswith("3"): suffix = ".SZ"
    elif raw_code.startswith("4") or raw_code.startswith("8"): suffix = ".BJ"
    else: suffix = ".SS"
    return f"{raw_code}{suffix}", raw_name

@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker, start, end):
    """
    完全同步 Deepnote 的特征工程逻辑
    """
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
        
        # 自动纠错后缀
        if df.empty or len(df) < 10:
            base_code = ticker.split('.')[0]
            current_suffix = '.' + ticker.split('.')[1]
            alt_suffix = '.SZ' if current_suffix == '.SS' else '.SS'
            alt_ticker = base_code + alt_suffix
            df = yf.download(alt_ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
            if not df.empty and len(df) > 10:
                ticker = alt_ticker

        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass 
            
        if len(df) < 60: return None, ticker # 同步 Deepnote 的数据长度检查

        if 'Close' not in df.columns: return None, ticker

        # --- 核心特征工程 (同步 Deepnote) ---
        data = df[['Close', 'High', 'Low', 'Volume']].copy()
        data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
        # 窗口大小由参数控制，默认20
        data['Volatility'] = data['Log_Ret'].rolling(window=20).std()
        
        # [A股特色] 加入量比因子 (虽然HMM暂未使用，但保持结构一致)
        data['Vol_Change'] = (data['Volume'] - data['Volume'].rolling(window=5).mean()) / data['Volume'].rolling(window=5).mean()
        
        data.dropna(inplace=True)
        return data, ticker
    except Exception as e:
        return None, ticker

# ==========================================
# 2. 模型训练 (同步 Random Seed)
# ==========================================
def train_bhmm(df, n_comps):
    scale = 100.0
    X = df[['Log_Ret', 'Volatility']].values * scale
    
    try:
        # === 关键修正: Random State 改为 88 ===
        model = GaussianHMM(n_components=n_comps, covariance_type="full", n_iter=1000, 
                           random_state=88, tol=0.01, min_covar=0.001)
        model.fit(X)
    except: return None

    hidden_states = model.predict(X)
    
    # 状态排序
    state_vol_means = [(i, X[hidden_states == i, 1].mean()) for i in range(n_comps)]
    sorted_stats = sorted(state_vol_means, key=lambda x: x[1])
    mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
    
    df['Regime'] = np.array([mapping[s] for s in hidden_states])
    
    # 贝叶斯后验
    state_means = np.array([df[df['Regime'] == i]['Log_Ret'].mean() for i in range(n_comps)])
    new_transmat = np.zeros_like(model.transmat_)
    for i in range(n_comps):
        for j in range(n_comps):
            new_transmat[mapping[i], mapping[j]] = model.transmat_[i, j]
            
    posterior_probs = model.predict_proba(X)
    sorted_probs = np.zeros_like(posterior_probs)
    for old_i, new_i in mapping.items():
        sorted_probs[:, new_i] = posterior_probs[:, old_i]
        
    next_day_probs = np.dot(sorted_probs, new_transmat)
    df['Bayes_Exp_Ret'] = np.dot(next_day_probs, state_means)
    
    return df

# ==========================================
# 3. 回测系统 (同步 Threshold)
# ==========================================
def backtest_strategy(df, cost):
    # === 关键修正: 阈值改为 0.0005 (5bps) ===
    # 过滤微小波动，减少来回打脸
    threshold = 0.0005 
    
    df['Signal'] = 0
    df.loc[df['Bayes_Exp_Ret'] > threshold, 'Signal'] = 1
    # 小于等于阈值时，Signal 为 0 (空仓)
    
    df['Position'] = df['Signal'].shift(1).fillna(0)
    t_cost = df['Position'].diff().abs() * cost
    
    df['Strategy_Ret'] = (df['Position'] * df['Log_Ret']) - t_cost
    df['Cum_Bench'] = (1 + df['Log_Ret']).cumprod()
    df['Cum_Strat'] = (1 + df['Strategy_Ret']).cumprod()
    
    total_ret = df['Cum_Strat'].iloc[-1] - 1
    annual_ret = (1 + total_ret) ** (252 / len(df)) - 1
    max_dd = ((df['Cum_Strat'] - df['Cum_Strat'].cummax()) / df['Cum_Strat'].cummax()).min()
    
    if df['Strategy_Ret'].std() != 0:
        sharpe = (df['Strategy_Ret'].mean() * 252) / (df['Strategy_Ret'].std() * np.sqrt(252))
    else: sharpe = 0
        
    return df, {"Total Return": total_ret, "CAGR": annual_ret, "Sharpe": sharpe, "Max Drawdown": max_dd}

# ==========================================
# 4. AI 投顾 (逻辑保持一致)
# ==========================================
def get_ai_advice(df, metrics, n_comps):
    last_regime = df['Regime'].iloc[-1]
    last_alpha = df['Bayes_Exp_Ret'].iloc[-1]
    
    advice = {"title": "", "color": "", "bg_color": "", "summary": "", "action": "", "risk_level": ""}
    
    # 阈值同步判断逻辑
    threshold = 0.0005

    if last_regime == 0: 
        advice['risk_level'] = "低 (Low)"
        if last_alpha > threshold:
            advice['title'] = "🟢 积极建仓机会 (Accumulate)"
            advice['color'] = "#00E676"
            advice['bg_color'] = "rgba(0, 230, 118, 0.1)"
            advice['summary'] = "低波动稳态，预期收益突破阈值 (Alpha > 5bps)。"
            advice['action'] = "建议：分批买入，持股待涨。"
        else:
            advice['title'] = "🟡 观望/防守 (Defensive)"
            advice['color'] = "#FFD600"
            advice['bg_color'] = "rgba(255, 214, 0, 0.1)"
            advice['summary'] = "波动率极低，但预期收益微弱 (Alpha < 5bps)。"
            advice['action'] = "建议：空仓观望，等待信号明确。"
            
    elif last_regime < n_comps - 1:
        advice['risk_level'] = "中 (Medium)"
        if last_alpha > threshold:
            advice['title'] = "🔵 趋势延续 (Trend)"
            advice['color'] = "#2962FF"
            advice['bg_color'] = "rgba(41, 98, 255, 0.1)"
            advice['summary'] = "趋势延续中，预期收益良好。"
            advice['action'] = "建议：继续持有。"
        else:
            advice['title'] = "🟠 减仓观望 (Reduce)"
            advice['color'] = "#FF9100"
            advice['bg_color'] = "rgba(255, 145, 0, 0.1)"
            advice['summary'] = "上涨动能衰竭。"
            advice['action'] = "建议：离场观望。"
    else:
        advice['risk_level'] = "高 (High)"
        advice['title'] = "🔴 极度风险预警 (Danger)"
        advice['color'] = "#FF1744"
        advice['bg_color'] = "rgba(255, 23, 68, 0.1)"
        advice['summary'] = "剧烈波动模式，暴跌风险高。"
        advice['action'] = "建议：清仓避险。"

    if last_alpha <= threshold: pos_sugg = "0%"
    elif last_regime == n_comps - 1: pos_sugg = "0-10%"
    else:
        base_pos = 50
        if last_regime == 0: base_pos += 30
        if last_alpha > 0.002: base_pos += 20
        pos_sugg = f"{min(base_pos, 100)}%"
    advice['position'] = pos_sugg
    return advice

# ==========================================
# 5. 主程序
# ==========================================
with st.sidebar:
    st.title("🇨🇳 A-Share Config")
    st.caption("参数已与 Deepnote 对齐")
    st.divider()

    with st.spinner("Connecting to AkShare..."):
        stock_list_df, is_online = get_all_a_share_list()
    
    target_ticker, target_name = None, None
    if is_online:
        selected = st.selectbox("代码/名称搜索", options=stock_list_df['Display'])
        if selected:
            c, n = selected.split(" | ")
            target_ticker, target_name = format_ticker_for_yfinance(c, n)
    else:
        mc = st.text_input("代码 (离线)", value="002340")
        if mc: target_ticker, target_name = format_ticker_for_yfinance(mc, mc)

    st.divider()
    # 默认值修改为 3状态, 3年回测
    n_components = st.slider("状态数 (Regimes)", 2, 4, 3)
    lookback_years = st.slider("回测年限", 1, 5, 3)
    
    # === 关键修正: 默认成本改为 10bps (Deepnote 为 0.001) ===
    trans_cost_bps = st.number_input("双边成本 (bps)", value=10, help="Deepnote 默认为 10bps (万10)") 
    transaction_cost = trans_cost_bps / 10000

    start_date = (datetime.now() - timedelta(days=365*lookback_years)).strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    run_btn = st.button("🚀 同步运算", type="primary", use_container_width=True)

st.title("🇨🇳 A-Share BHMM Analytics")
if run_btn and target_ticker:
    with st.spinner(f"正在分析 {target_name}..."):
        df, final_ticker = get_data(target_ticker, start_date, end_date)
        if df is None: st.error("No Data"); st.stop()
        
        df = train_bhmm(df, n_components)
        if df is None: st.stop()
        
        df, metrics = backtest_strategy(df, transaction_cost)
        ai_advice = get_ai_advice(df, metrics, n_components)
        
        # --- UI ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("累计收益", f"{metrics['Total Return']*100:.1f}%")
        c2.metric("年化收益", f"{metrics['CAGR']*100:.1f}%")
        c3.metric("夏普比率", f"{metrics['Sharpe']:.2f}")
        c4.metric("最大回撤", f"{metrics['Max Drawdown']*100:.1f}%")
        
        st.markdown(f"""
        <div style="background:{ai_advice['bg_color']}; padding:20px; border-radius:10px; border-left:5px solid {ai_advice['color']}; margin:20px 0;">
            <h3 style="color:{ai_advice['color']}; margin:0;">{ai_advice['title']}</h3>
            <p style="color:#ccc; margin-top:10px;">{ai_advice['summary']}</p>
            <div style="display:flex; justify-content:space-between; margin-top:15px; font-weight:bold;">
                <span style="color:#fff;">建议操作: {ai_advice['action']}</span>
                <span style="color:{ai_advice['color']};">仓位: {ai_advice['position']}</span>
            </div>
            <div style="margin-top:5px; font-size:0.8em; color:#888;">
                 Deepnote 对齐参数: Seed=88, Threshold=5bps, Cost={trans_cost_bps}bps
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Regimes", "Equity"])
        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            colors = ['#00E676', '#FFD600', '#FF1744', '#AA00FF']
            for i in range(n_components):
                mask = df['Regime'] == i
                if mask.any():
                    fig.add_trace(go.Scatter(x=df.index[mask], y=df['Close'][mask], mode='markers', marker=dict(size=4, color=colors[i%4]), name=f"State {i}"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], line=dict(color='rgba(255,255,255,0.2)', width=1), showlegend=False), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='rgba(255,255,255,0.3)', name="Volume"), row=2, col=1)
            fig.update_layout(template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=df.index, y=df['Cum_Bench'], name="基准", line=dict(color='gray', dash='dot')))
            fig_eq.add_trace(go.Scatter(x=df.index, y=df['Cum_Strat'], name="BHMM 策略", line=dict(color='#FF5252', width=2)))
            fig_eq.update_layout(template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_eq, use_container_width=True)
elif run_btn:
    st.warning("Please select a valid ticker.")
else:
    st.info("👈 请在侧边栏点击同步运算")
