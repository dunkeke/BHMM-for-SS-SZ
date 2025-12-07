import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import akshare as ak  # 引入 AkShare 获取全市场列表
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

# ==========================================
# 0. 页面配置与高级 UI
# ==========================================
st.set_page_config(
    page_title="BHMM A-Share Sniper",
    page_icon="🇨🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore")

# 引入之前的“彭博风”暗黑霓虹 CSS
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
    h1, h2, h3 { font-family: 'SF Pro Display', sans-serif; letter-spacing: -0.5px; }
    div.stButton > button {
        background: linear-gradient(90deg, #D32F2F 0%, #FF5252 100%); /* A股红 */
        color: white; border: none; font-weight: 600;
    }
    /* AI Advice Box */
    .ai-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        margin-bottom: 20px;
        border-left: 5px solid;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. A股 核心数据引擎 (AkShare + YFinance)
# ==========================================

@st.cache_data(ttl=24*3600)  # 缓存 24 小时
def get_all_a_share_list():
    """使用 AkShare 获取全市场实时行情数据"""
    try:
        df = ak.stock_zh_a_spot_em()
        df = df[['代码', '名称']]
        df['Display'] = df['代码'] + " | " + df['名称']
        return df, True
    except Exception as e:
        fallback_data = {
            "代码": ["600519", "300750", "000001", "000858"],
            "名称": ["贵州茅台", "宁德时代", "平安银行", "五粮液"],
            "Display": ["600519 | 贵州茅台", "300750 | 宁德时代", "000001 | 平安银行", "000858 | 五粮液"]
        }
        return pd.DataFrame(fallback_data), False

@st.cache_data(ttl=3600)
def format_ticker_for_yfinance(raw_code, raw_name="Unknown"):
    """代码转 YFinance 格式"""
    raw_code = str(raw_code).strip()
    if raw_code.startswith("6"): suffix = ".SS"
    elif raw_code.startswith("9"): suffix = ".SS"
    elif raw_code.startswith("0") or raw_code.startswith("3"): suffix = ".SZ"
    elif raw_code.startswith("4") or raw_code.startswith("8"): suffix = ".BJ"
    else: suffix = ".SS"
    return f"{raw_code}{suffix}", raw_name

@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker, start, end):
    """获取数据，包含自动纠错机制"""
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
        
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
            
        if len(df) < 252: return None, ticker
        if 'Close' not in df.columns: return None, ticker

        data = df[['Close', 'Volume']].copy()
        data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Volatility'] = data['Log_Ret'].rolling(window=20).std()
        data.dropna(inplace=True)
        return data, ticker
    except Exception as e:
        return None, ticker

# ==========================================
# 2. 核心 HMM 逻辑
# ==========================================
def train_bhmm(df, n_comps):
    scale = 100.0
    X = df[['Log_Ret', 'Volatility']].values * scale
    
    try:
        model = GaussianHMM(n_components=n_comps, covariance_type="full", n_iter=1000, 
                           random_state=42, tol=0.01, min_covar=0.001)
        model.fit(X)
    except: return None

    hidden_states = model.predict(X)
    
    # 排序：0=低波, N=高波
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

def backtest_strategy(df, cost):
    threshold = 0.0000 
    df['Signal'] = 0
    df.loc[df['Bayes_Exp_Ret'] > threshold, 'Signal'] = 1
    
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
# 3. 新增：AI 智能投顾模块
# ==========================================
def get_ai_advice(df, metrics, n_comps):
    """
    基于量化结果生成自然语言投资建议
    """
    last_regime = df['Regime'].iloc[-1]
    last_alpha = df['Bayes_Exp_Ret'].iloc[-1]
    last_vol = df['Volatility'].iloc[-1]
    
    advice = {
        "title": "", "color": "", "bg_color": "",
        "summary": "", "action": "", "risk_level": ""
    }
    
    # 1. 判断核心立场 (基于 Regime 和 Alpha)
    # Regime 0: 低波 (通常是筑底或缓慢爬升)
    # Regime N: 高波 (通常是顶部或崩盘)
    
    if last_regime == 0: # 低波动状态
        advice['risk_level'] = "低 (Low)"
        if last_alpha > 0:
            advice['title'] = "🟢 积极建仓机会 (Accumulate)"
            advice['color'] = "#00E676" # Green
            advice['bg_color'] = "rgba(0, 230, 118, 0.1)"
            advice['summary'] = "市场处于低波动稳态，且贝叶斯预期收益为正。这通常是主力资金吸筹或趋势启动初期的特征。"
            advice['action'] = "建议：分批买入，持股待涨。适合重仓布局。"
        else:
            advice['title'] = "🟡 观望/防守 (Defensive)"
            advice['color'] = "#FFD600" # Yellow
            advice['bg_color'] = "rgba(255, 214, 0, 0.1)"
            advice['summary'] = "市场波动率极低，呈现横盘死水状态，且预期收益微弱。方向不明。"
            advice['action'] = "建议：保持空仓或极轻仓，等待趋势突破信号。"
            
    elif last_regime < n_comps - 1: # 中间状态 (趋势延续)
        advice['risk_level'] = "中 (Medium)"
        if last_alpha > 0:
            advice['title'] = "🔵 趋势延续 (Trend Following)"
            advice['color'] = "#2962FF" # Blue
            advice['bg_color'] = "rgba(41, 98, 255, 0.1)"
            advice['summary'] = "市场处于良性波动区间，上涨趋势未被破坏。"
            advice['action'] = "建议：继续持有，可适当设置移动止盈。"
        else:
            advice['title'] = "🟠 减仓观望 (Reduce)"
            advice['color'] = "#FF9100" # Orange
            advice['bg_color'] = "rgba(255, 145, 0, 0.1)"
            advice['summary'] = "虽然并未进入极度恐慌，但上涨动能衰竭，预期转负。"
            advice['action'] = "建议：降低仓位，锁定利润。"
            
    else: # 高波动状态 (Regime N)
        advice['risk_level'] = "高 (High)"
        advice['title'] = "🔴 极度风险预警 (Danger)"
        advice['color'] = "#FF1744" # Red
        advice['bg_color'] = "rgba(255, 23, 68, 0.1)"
        advice['summary'] = "市场进入剧烈波动模式（Regime Max）。根据历史回测，此状态下暴跌概率极高，风险收益比极差。"
        advice['action'] = "建议：立即清仓或利用衍生品对冲。现金为王。"

    # 2. 凯利公式仓位建议 (简化版)
    # 假设赔率 b=1, 胜率 p 估算为 Alpha 强度的映射 (仅供参考)
    # 这里我们用直观的逻辑代替复杂公式
    if last_alpha <= 0:
        pos_sugg = "0%"
    elif last_regime == n_comps - 1:
        pos_sugg = "0-10% (彩票仓)"
    else:
        # Alpha 越高，仓位越重；波动率越低，仓位越重
        base_pos = 50
        if last_regime == 0: base_pos += 30
        if last_alpha > 0.001: base_pos += 20
        pos_sugg = f"{min(base_pos, 100)}%"
        
    advice['position'] = pos_sugg
    return advice

# ==========================================
# 4. 主面板逻辑
# ==========================================
with st.sidebar:
    st.title("🇨🇳 A-Share Config")
    st.divider()

    # 搜索逻辑
    with st.spinner("Load Market List..."):
        stock_list_df, is_online = get_all_a_share_list()
    
    target_ticker = None
    target_name = None

    if is_online:
        selected_option = st.selectbox("代码/名称搜索", options=stock_list_df['Display'])
        if selected_option:
            code_part, name_part = selected_option.split(" | ")
            target_ticker, target_name = format_ticker_for_yfinance(code_part, name_part)
            st.info(f"已锁定: {name_part}")
    else:
        manual_code = st.text_input("输入代码 (离线模式)", value="002340")
        if manual_code:
            target_ticker, target_name = format_ticker_for_yfinance(manual_code, manual_code)

    st.divider()
    n_components = st.slider("状态数 (Regimes)", 2, 4, 3)
    lookback_years = st.slider("回测年限", 1, 5, 3)
    transaction_cost = st.number_input("成本 (bps)", value=5) / 10000
    
    start_date = (datetime.now() - timedelta(days=365*lookback_years)).strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    run_btn = st.button("🚀 开始量化分析", type="primary", use_container_width=True)

st.title("🇨🇳 A-Share BHMM Analytics")
st.markdown("利用贝叶斯隐马尔可夫模型识别 **A股资金风格 (Regimes)**")

if run_btn and target_ticker:
    with st.spinner(f"正在分析: {target_name} ..."):
        # 1. 获取数据
        df, final_ticker = get_data(target_ticker, start_date, end_date)
        if df is None:
            st.error("无法获取数据，请检查代码或网络。")
            st.stop()
            
        # 2. 训练
        df = train_bhmm(df, n_components)
        if df is None: st.stop()
        
        # 3. 回测
        df, metrics = backtest_strategy(df, transaction_cost)
        
        # 4. 生成 AI 建议
        ai_advice = get_ai_advice(df, metrics, n_components)
        
        # --- UI 展示 ---
        
        # A. 策略指标
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("累计收益", f"{metrics['Total Return']*100:.1f}%", delta_color="normal" if metrics['Total Return']>0 else "inverse")
        c2.metric("年化收益", f"{metrics['CAGR']*100:.1f}%")
        c3.metric("夏普比率", f"{metrics['Sharpe']:.2f}")
        c4.metric("最大回撤", f"{metrics['Max Drawdown']*100:.1f}%")
        
        # B. AI 投资建议卡片 (核心更新)
        st.markdown(f"""
        <div style="
            background-color: {ai_advice['bg_color']};
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid {ai_advice['color']};
            margin: 25px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <h3 style="margin-top:0; color: {ai_advice['color']}; display: flex; align-items: center;">
                {ai_advice['title']}
            </h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
                <div>
                    <strong style="color: #ddd;">📈 市场状态分析:</strong>
                    <p style="color: #bbb; margin-top: 5px;">{ai_advice['summary']}</p>
                </div>
                <div>
                    <strong style="color: #ddd;">⚡ 操作建议:</strong>
                    <p style="color: #fff; font-weight: 500; margin-top: 5px;">{ai_advice['action']}</p>
                </div>
            </div>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <div style="display: flex; justify-content: space-between; font-family: 'Roboto Mono';">
                <span>建议参考仓位: <strong style="color: {ai_advice['color']}">{ai_advice['position']}</strong></span>
                <span>当前风险等级: {ai_advice['risk_level']}</span>
                <span>次日预期 Alpha: {df['Bayes_Exp_Ret'].iloc[-1]*100:.3f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # C. 图表
        tab1, tab2 = st.tabs(["📊 价格与风格", "💰 净值曲线"])
        
        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            colors = ['#00E676', '#FFD600', '#FF1744', '#AA00FF']
            for i in range(n_components):
                mask = df['Regime'] == i
                if mask.any():
                    fig.add_trace(go.Scatter(x=df.index[mask], y=df['Close'][mask], mode='markers', 
                                           marker=dict(size=4, color=colors[i%4]), name=f"Regime {i}"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], line=dict(color='rgba(255,255,255,0.2)', width=1), showlegend=False), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='rgba(255,255,255,0.3)', name="Volume"), row=2, col=1)
            fig.update_layout(template="plotly_dark", height=500, margin=dict(t=20, b=20), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=df.index, y=df['Cum_Bench'], name="基准", line=dict(color='gray', dash='dot')))
            fig_eq.add_trace(go.Scatter(x=df.index, y=df['Cum_Strat'], name="BHMM 策略", line=dict(color='#FF5252', width=2)))
            fig_eq.update_layout(template="plotly_dark", height=450, margin=dict(t=20, b=20), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_eq, use_container_width=True)

elif run_btn:
    st.warning("请先配置股票代码。")
else:
    st.info("👈 请在侧边栏开始分析")
