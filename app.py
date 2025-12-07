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
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. A股 核心数据引擎 (AkShare + YFinance)
# ==========================================

@st.cache_data(ttl=24*3600)  # 缓存 24 小时，避免每次刷新都请求接口
def get_all_a_share_list():
    """
    使用 AkShare 获取全市场实时行情数据，仅提取代码和名称。
    """
    try:
        # 获取 A 股实时行情 (包含代码、名称)
        # 这是一个轻量级接口，速度较快
        df = ak.stock_zh_a_spot_em()
        df = df[['代码', '名称']]
        
        # 格式化显示列： "600519 | 贵州茅台"
        df['Display'] = df['代码'] + " | " + df['名称']
        return df
    except Exception as e:
        # 如果接口挂了，返回一个保底的小列表
        fallback_data = {
            "代码": ["600519", "300750", "000001", "000858"],
            "名称": ["贵州茅台", "宁德时代", "平安银行", "五粮液"],
            "Display": ["600519 | 贵州茅台", "300750 | 宁德时代", "000001 | 平安银行", "000858 | 五粮液"]
        }
        return pd.DataFrame(fallback_data)

@st.cache_data(ttl=3600)
def format_ticker_for_yfinance(raw_code, raw_name):
    """
    将 AkShare 的 6 位纯数字代码转换为 YFinance 需要的格式 (.SS/.SZ)
    """
    # 规则判断
    if raw_code.startswith("6"):
        suffix = ".SS" # 沪市主板/科创板
    elif raw_code.startswith("9"):
        suffix = ".SS" # 沪市B股 (极少用)
    elif raw_code.startswith("0") or raw_code.startswith("3"):
        suffix = ".SZ" # 深市/创业板
    elif raw_code.startswith("4") or raw_code.startswith("8"):
        suffix = ".BJ" # 北交所 (注意：YFinance 对北交所支持较差，可能会获取失败)
    else:
        suffix = ".SS" # 默认回退
        
    return f"{raw_code}{suffix}", raw_name

@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker, start, end):
    """获取数据，包含自动纠错机制"""
    try:
        # 第一次尝试
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
        
        # 自动纠错逻辑：如果 .SS 没数据，尝试切换 .SZ
        if df.empty or len(df) < 10:
            base_code = ticker.split('.')[0]
            current_suffix = '.' + ticker.split('.')[1]
            # 简单的互换逻辑
            alt_suffix = '.SZ' if current_suffix == '.SS' else '.SS'
            alt_ticker = base_code + alt_suffix
            
            df = yf.download(alt_ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
            if not df.empty and len(df) > 10:
                ticker = alt_ticker # 更新成功的 ticker

        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass 
            
        if len(df) < 252: return None, ticker

        if 'Close' not in df.columns: return None, ticker

        data = df[['Close', 'Volume']].copy()
        data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
        # A股的波动率计算窗口可以稍微短一点，反应更快
        data['Volatility'] = data['Log_Ret'].rolling(window=20).std()
        
        data.dropna(inplace=True)
        return data, ticker
    except Exception as e:
        return None, ticker

# ==========================================
# 2. 侧边栏：动态搜索配置
# ==========================================
with st.sidebar:
    st.title("🇨🇳 A-Share Config")
    st.caption("中国 A 股全市场扫描")
    st.divider()

    st.subheader("1. 标的搜索 (Target)")
    
    # 获取全市场列表 (带缓存)
    with st.spinner("正在加载 A 股全市场列表..."):
        stock_list_df = get_all_a_share_list()
    
    # 使用 Selectbox 实现搜索功能
    # Streamlit 的 Selectbox 原生支持输入文字进行过滤，非常适合这个场景
    selected_option = st.selectbox(
        "输入代码或名称搜索 (支持 5000+ 只股票)",
        options=stock_list_df['Display'],
        index=0, # 默认选中第一个
        help="数据来源: AkShare (实时更新)"
    )
    
    # 解析用户的选择
    if selected_option:
        # split "600519 | 贵州茅台"
        code_part = selected_option.split(" | ")[0]
        name_part = selected_option.split(" | ")[1]
        
        # 转换为 YF 格式
        target_ticker, target_name = format_ticker_for_yfinance(code_part, name_part)
        st.info(f"已锁定: **{name_part}** ({target_ticker})")
    
    st.divider()

    st.subheader("2. 模型参数 (HMM)")
    n_components = st.slider("市场状态数 (Regimes)", 2, 4, 3)
    lookback_years = st.slider("回测年限", 1, 5, 3)
    
    trans_cost_bps = st.number_input("双边交易成本 (bps)", value=5) 
    transaction_cost = trans_cost_bps / 10000

    start_date = (datetime.now() - timedelta(days=365*lookback_years)).strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    st.divider()
    run_btn = st.button("🚀 开始量化分析 (Analyze)", type="primary", use_container_width=True)

# ==========================================
# 3. 核心 HMM 逻辑 (复用并微调)
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
    
    # A股逻辑：按波动率排序，0=低波震荡(往往是建仓期), N=高波(往往是顶部或崩盘)
    state_vol_means = [(i, X[hidden_states == i, 1].mean()) for i in range(n_comps)]
    sorted_stats = sorted(state_vol_means, key=lambda x: x[1])
    mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
    
    df['Regime'] = np.array([mapping[s] for s in hidden_states])
    
    # 计算贝叶斯后验预期收益
    state_means = np.array([df[df['Regime'] == i]['Log_Ret'].mean() for i in range(n_comps)])
    
    # 重构转移矩阵
    new_transmat = np.zeros_like(model.transmat_)
    for i in range(n_comps):
        for j in range(n_comps):
            new_transmat[mapping[i], mapping[j]] = model.transmat_[i, j]
            
    # 预测下一日收益
    posterior_probs = model.predict_proba(X)
    sorted_probs = np.zeros_like(posterior_probs)
    for old_i, new_i in mapping.items():
        sorted_probs[:, new_i] = posterior_probs[:, old_i]
        
    next_day_probs = np.dot(sorted_probs, new_transmat)
    df['Bayes_Exp_Ret'] = np.dot(next_day_probs, state_means)
    
    return df

def backtest_strategy(df, cost):
    # A股做空限制：
    # 虽然A股有融券，但为简化模型，我们假设这是一个“仅做多 (Long Only)”策略
    # 逻辑：预期收益 > 阈值 买入/持有，否则 空仓
    
    threshold = 0.0000 # 只要预期为正就持有
    
    df['Signal'] = 0
    # 只有做多逻辑
    df.loc[df['Bayes_Exp_Ret'] > threshold, 'Signal'] = 1
    # df.loc[df['Bayes_Exp_Ret'] < -threshold, 'Signal'] = -1 # A股如果不做融券，这里可以注释掉
    
    df['Position'] = df['Signal'].shift(1).fillna(0)
    
    # 交易成本
    trades = df['Position'].diff().abs()
    t_cost = trades * cost
    
    df['Strategy_Ret'] = (df['Position'] * df['Log_Ret']) - t_cost
    df['Cum_Bench'] = (1 + df['Log_Ret']).cumprod()
    df['Cum_Strat'] = (1 + df['Strategy_Ret']).cumprod()
    
    # 指标计算
    total_ret = df['Cum_Strat'].iloc[-1] - 1
    annual_ret = (1 + total_ret) ** (252 / len(df)) - 1
    max_dd = ((df['Cum_Strat'] - df['Cum_Strat'].cummax()) / df['Cum_Strat'].cummax()).min()
    
    if df['Strategy_Ret'].std() != 0:
        sharpe = (df['Strategy_Ret'].mean() * 252) / (df['Strategy_Ret'].std() * np.sqrt(252))
    else: sharpe = 0
        
    return df, {"Total Return": total_ret, "CAGR": annual_ret, "Sharpe": sharpe, "Max Drawdown": max_dd}

# ==========================================
# 4. 主面板逻辑
# ==========================================
st.title("🇨🇳 A-Share BHMM Analytics")
st.markdown("利用贝叶斯隐马尔可夫模型识别 **A股资金风格 (Regimes)**")

if run_btn and target_ticker:
    with st.spinner(f"正在连接交易所数据: {target_name} ({target_ticker}) ..."):
        # 1. 获取数据
        df, final_ticker = get_data(target_ticker, start_date, end_date)
        
        if df is None:
            st.error(f"无法获取 {target_ticker} 的数据。可能是代码错误或网络问题。")
            st.stop()
            
        st.success(f"成功获取数据: {len(df)} 交易日 (Ticker: {final_ticker})")
        
        # 2. 训练模型
        with st.spinner("正在拟合高斯混合模型 (GMM) ..."):
            df = train_bhmm(df, n_components)
            if df is None:
                st.error("模型训练发散，请尝试调整参数。")
                st.stop()
        
        # 3. 回测
        df, metrics = backtest_strategy(df, transaction_cost)
        
        # 4. 展示结果
        
        # --- 顶部卡片 ---
        st.markdown("### 📊 策略概览")
        c1, c2, c3, c4 = st.columns(4)
        
        # 根据收益变色
        ret_color = "normal" if metrics['Total Return'] > 0 else "inverse"
        c1.metric("累计收益 (Total Return)", f"{metrics['Total Return']*100:.1f}%", delta_color=ret_color)
        c2.metric("年化收益 (CAGR)", f"{metrics['CAGR']*100:.1f}%")
        c3.metric("夏普比率 (Sharpe)", f"{metrics['Sharpe']:.2f}")
        c4.metric("最大回撤 (Max DD)", f"{metrics['Max Drawdown']*100:.1f}%")
        
        # --- 信号卡片 ---
        last_regime = df['Regime'].iloc[-1]
        last_alpha = df['Bayes_Exp_Ret'].iloc[-1]
        
        # A股风格解读
        regime_desc = {
            0: "📉 低波震荡 (往往是筑底/横盘)",
            1: ⚖️ 中波趋势 (正常的上涨/下跌)",
            2: "🌋 高波剧烈 (顶部狂热或崩盘)",
            3: "🌪️ 极端波动"
        }
        status_text = regime_desc.get(last_regime, f"State {last_regime}")
        
        st.markdown(f"""
        <div style="background: rgba(41, 98, 255, 0.1); border-radius: 8px; padding: 20px; border-left: 5px solid #2962FF; margin: 20px 0;">
            <h3 style="margin:0; color: #2962FF;">当前市场状态: {status_text}</h3>
            <p style="margin: 5px 0 0 0; color: #aaa;">贝叶斯预判次日 Alpha: <strong style="color: white;">{last_alpha*100:.3f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # --- 图表区域 ---
        tab1, tab2 = st.tabs(["📈 价格与风格 (Regimes)", "💰 净值曲线 (Equity)"])
        
        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            
            # K线/收盘价 + 颜色点
            # A股习惯：红涨绿跌，但这里我们用颜色区分Regime
            colors = ['#00E676', '#FFD600', '#FF1744', '#AA00FF'] # 绿(稳), 黄(变), 红(危)
            
            for i in range(n_components):
                mask = df['Regime'] == i
                if mask.any():
                    fig.add_trace(go.Scatter(
                        x=df.index[mask], y=df['Close'][mask],
                        mode='markers', marker=dict(size=4, color=colors[i%4]),
                        name=f"Regime {i}"
                    ), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], line=dict(color='rgba(255,255,255,0.2)', width=1), showlegend=False), row=1, col=1)
            
            # 成交量 (Volume) - A股分析很重要
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='rgba(255,255,255,0.3)', name="Volume"), row=2, col=1)
            
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=20,b=20), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=df.index, y=df['Cum_Bench'], name="基准 (买入持有)", line=dict(color='gray', dash='dot')))
            fig_eq.add_trace(go.Scatter(x=df.index, y=df['Cum_Strat'], name="BHMM 择时策略", line=dict(color='#FF5252', width=2)))
            
            # 标记买卖点
            # 只有当持仓从0变1 (买入) 或 1变0 (卖出)
            buys = df[(df['Position']==1) & (df['Position'].shift(1)==0)]
            sells = df[(df['Position']==0) & (df['Position'].shift(1)==1)]
            
            fig_eq.add_trace(go.Scatter(x=buys.index, y=df.loc[buys.index, 'Cum_Strat'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='red'), name="买入"))
            fig_eq.add_trace(go.Scatter(x=sells.index, y=df.loc[sells.index, 'Cum_Strat'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='green'), name="卖出"))
            
            fig_eq.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,t=20,b=20), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_eq, use_container_width=True)

elif run_btn and not target_ticker:
    st.warning("⚠️ 请先在侧边栏输入有效的股票代码或选择蓝筹股。")

else:
    # 空闲状态显示
    st.info("👈 请在左侧侧边栏配置股票和参数，然后点击 '🚀 开始量化分析'")
    
    with st.expander("📖 搜索提示"):
        st.markdown("""
        **AkShare 全量数据支持**：
        - 下拉框已包含全市场 5000+ 股票。
        - 可以在下拉框中**直接输入**代码（如 `600`）或中文（如 `茅台`）进行模糊筛选。
        """)