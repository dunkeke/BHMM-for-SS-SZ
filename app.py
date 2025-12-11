import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

# 尝试引入鲁棒性模块
try:
    from robustness import RobustnessLab
except ImportError:
    st.error("⚠️ 缺少 robustness.py 文件，无法运行鲁棒性测试模块。")

warnings.filterwarnings("ignore")

# ==========================================
# 0. A股配置与板块数据 (Sector Data)
# ==========================================
st.set_page_config(page_title="A-Share Alpha Scanner", layout="wide", page_icon="🇨🇳")

# A股核心板块成分股 (精选龙头，用于扫描演示)
SECTORS = {
    "茅指数 (核心资产)": {
        "贵州茅台": "600519.SS", "五粮液": "000858.SZ", "招商银行": "600036.SS", 
        "中国平安": "601318.SS", "恒瑞医药": "600276.SS", "美的集团": "000333.SZ"
    },
    "宁组合 (新能源/科技)": {
        "宁德时代": "300750.SZ", "比亚迪": "002594.SZ", "隆基绿能": "601012.SS", 
        "阳光电源": "300274.SZ", "立讯精密": "002475.SZ", "北方华创": "002371.SZ"
    },
    "中特估 (高股息)": {
        "长江电力": "600900.SS", "中国神华": "601088.SS", "中国移动": "600941.SS", 
        "农业银行": "601288.SS", "陕西煤业": "601225.SS", "大秦铁路": "601006.SS"
    }
}

# A股费率设置 (印花税+佣金+滑点，保守估计万5)
ASHARE_COST = 0.0005

# ==========================================
# PART 1: 策略适配 (Long-Only Adapter)
# ==========================================

class StrategyBase:
    def generate_signals(self, df): raise NotImplementedError

class HMMAdaptiveAshare(StrategyBase):
    """
    [A股特供版] HMM 自适应策略
    特点: 
    1. 只能做多 (Long Only): 信号 -1 强制转为 0 (空仓)
    2. 贝叶斯后验优化
    """
    def __init__(self, n_components=3, iter_num=1000, window_size=21, threshold=0.0003, **kwargs):
        self.n_components = n_components
        self.iter_num = iter_num
        self.window_size = window_size
        self.threshold = threshold

    def generate_signals(self, df):
        df = df.copy()
        # 基础特征工程
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Log_Ret'].rolling(window=self.window_size).std()
        
        # A股特色因子：量比 (成交量/5日均量) - 辅助判断活跃度
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()
        
        df.dropna(inplace=True)
        if len(df) < 60: return df # A股新股数据保护
        
        # HMM 训练
        X = df[['Log_Ret', 'Volatility']].values * 100.0
        try:
            model = GaussianHMM(n_components=self.n_components, covariance_type="full", n_iter=self.iter_num, random_state=88, tol=0.01, min_covar=0.01)
            model.fit(X)
        except: return df
        
        # 状态排序 (按波动率从小到大: 0=低波/吸筹, N=高波/出货)
        hidden_states = model.predict(X)
        state_vol_means = [X[hidden_states == i, 1].mean() for i in range(self.n_components)]
        sorted_stats = sorted(list(enumerate(state_vol_means)), key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
        
        # 贝叶斯推断
        posterior_probs = model.predict_proba(X)
        sorted_probs = np.zeros_like(posterior_probs)
        for old_i, new_i in mapping.items():
            sorted_probs[:, new_i] = posterior_probs[:, old_i]
            
        df['Regime'] = np.array([mapping[s] for s in hidden_states])
        
        # 计算各状态历史平均收益 (Priors)
        state_means = []
        for i in range(self.n_components):
            mean_ret = df[df['Regime'] == i]['Log_Ret'].mean()
            state_means.append(mean_ret)
            
        # 转移矩阵映射与预测
        new_transmat = np.zeros_like(model.transmat_)
        for i in range(self.n_components):
            for j in range(self.n_components):
                new_transmat[mapping[i], mapping[j]] = model.transmat_[i, j]
                
        next_probs = np.dot(sorted_probs, new_transmat)
        df['Bayes_Exp_Ret'] = np.dot(next_probs, state_means)
        
        # --- A股 信号生成逻辑 (Long Only) ---
        df['Signal'] = 0
        # 买入条件: 预期收益 > 阈值
        df.loc[df['Bayes_Exp_Ret'] > self.threshold, 'Signal'] = 1
        # 卖出条件: 预期收益 < -阈值 (转为0，即空仓)
        df.loc[df['Bayes_Exp_Ret'] < -self.threshold, 'Signal'] = 0 
        
        return df

# ==========================================
# PART 2: 扫描器引擎 (Scanner Engine)
# ==========================================

def run_scanner(sector_dict, start_date, end_date):
    """
    全市场扫描核心逻辑
    遍历板块个股 -> 训练HMM -> 提取当前状态与预期收益 -> 排序
    """
    results = []
    
    # 创建进度条
    progress_bar = st.progress(0)
    total = len(sector_dict)
    
    for idx, (name, ticker) in enumerate(sector_dict.items()):
        try:
            # 下载数据
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            if not df.empty and len(df) > 100:
                # 运行 HMM
                strat = HMMAdaptiveAshare(window_size=20)
                df_res = strat.generate_signals(df)
                
                if 'Regime' in df_res.columns:
                    last_row = df_res.iloc[-1]
                    
                    # 评分逻辑: 预期收益 * 10000 (bps)
                    score = last_row['Bayes_Exp_Ret'] * 10000
                    
                    # 状态解读
                    regime = int(last_row['Regime'])
                    status = "🟢 底部/拉升" if regime == 0 else ("🔴 顶部/巨震" if regime == 2 else "🟡 震荡/中继")
                    
                    results.append({
                        "名称": name,
                        "代码": ticker,
                        "当前价格": f"{last_row['Close']:.2f}",
                        "HMM状态": status,
                        "RegimeID": regime,
                        "预期Alpha (bps)": f"{score:.2f}",
                        "Raw_Alpha": last_row['Bayes_Exp_Ret'],
                        "建议": "💪 强力买入" if (regime == 0 and score > 5) else ("👀 关注" if score > 0 else "🛑 观望")
                    })
        except Exception as e:
            pass
            
        progress_bar.progress((idx + 1) / total)
        
    return pd.DataFrame(results)

# ==========================================
# PART 3: 回测引擎 (A股 T+1 适配)
# ==========================================

class AshareBacktestEngine:
    def __init__(self, initial_capital=100000, transaction_cost=ASHARE_COST):
        self.initial_capital = initial_capital
        self.cost = transaction_cost

    def run(self, df):
        df = df.copy()
        # T+1 模拟: T日信号，T+1日执行
        # Position 代表 T+1 日持仓
        df['Position'] = df['Signal'].shift(1).fillna(0)
        
        # 交易发生时刻 (仓位变动)
        trades = df['Position'].diff().abs().fillna(0)
        fees = trades * self.cost
        
        # 策略收益 (A股没有做空收益，Position只能是0或1)
        df['Strategy_Ret'] = (df['Position'] * df['Log_Ret']) - fees
        
        df['Equity_Curve'] = self.initial_capital * (1 + df['Strategy_Ret']).cumprod()
        df['Benchmark_Curve'] = self.initial_capital * (1 + df['Log_Ret']).cumprod()
        return df

# ==========================================
# PART 4: Streamlit UI
# ==========================================

st.title("🇨🇳 A-Share Quant Lab: HMM 选股与择时")

# 侧边栏模式选择
mode = st.sidebar.radio("功能模式", ["📡 全市场扫描 (Scanner)", "📈 单标的深度分析 (Deep Dive)", "🛡️ 鲁棒性测试 (Robustness)"])

if mode == "📡 全市场扫描 (Scanner)":
    st.header("🔍 HMM 智能选股器 (Smart Scanner)")
    st.info("原理：对板块内所有股票进行实时 HMM 建模，寻找处于 **'Regime 0 (低波吸筹)'** 且 **'贝叶斯预期收益 > 0'** 的标的。")
    
    selected_sector = st.selectbox("选择扫描赛道", list(SECTORS.keys()))
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🚀 开始扫描", type="primary"):
            with st.spinner(f"正在扫描 {selected_sector} 核心资产..."):
                scan_df = run_scanner(
                    SECTORS[selected_sector], 
                    datetime.now() - timedelta(days=365*2), 
                    datetime.now()
                )
                
                if not scan_df.empty:
                    # 排序：优先展示买入建议，其次按预期收益排序
                    scan_df = scan_df.sort_values(by="Raw_Alpha", ascending=False)
                    
                    # 样式优化
                    st.success(f"扫描完成！共分析 {len(scan_df)} 只个股。")
                    
                    # 高亮展示 Top 3
                    top_picks = scan_df.head(3)
                    st.subheader("🏆 今日首选 (Top Picks)")
                    cols = st.columns(3)
                    for i, row in enumerate(top_picks.to_dict('records')):
                        with cols[i]:
                            st.metric(
                                label=f"{row['名称']} ({row['HMM状态']})",
                                value=row['当前价格'],
                                delta=f"Alpha: {row['预期Alpha (bps)']} bps"
                            )
                    
                    st.subheader("📋 完整榜单")
                    # 展示表格 (隐藏 Raw_Alpha)
                    st.dataframe(
                        scan_df.drop(columns=['Raw_Alpha', 'RegimeID']),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("数据获取失败，请检查网络或稍后重试。")

elif mode == "📈 单标的深度分析 (Deep Dive)":
    st.sidebar.markdown("---")
    # 允许用户输入自定义代码
    ticker_input = st.sidebar.text_input("输入 A 股代码 (例如 600519)", value="600519")
    
    # 自动补全后缀逻辑
    if not (ticker_input.endswith(".SS") or ticker_input.endswith(".SZ")):
        if ticker_input.startswith("6"): ticker_input += ".SS"
        else: ticker_input += ".SZ"
    
    st.header(f"📊 深度分析: {ticker_input}")
    
    if st.sidebar.button("运行分析"):
        start_d = datetime.now() - timedelta(days=365*3)
        end_d = datetime.now()
        
        df = yf.download(ticker_input, start=start_d, end=end_d, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        if not df.empty:
            strat = HMMAdaptiveAshare()
            df_res = strat.generate_signals(df)
            
            # 展示最新信号
            last = df_res.iloc[-1]
            col1, col2, col3 = st.columns(3)
            col1.metric("当前状态 (Regime)", f"{int(last['Regime'])}")
            col2.metric("贝叶斯预期收益", f"{last['Bayes_Exp_Ret']*100:.4f}%")
            col3.metric("建议仓位", "🟢 满仓" if last['Signal']==1 else "⚪ 空仓")
            
            # 回测
            engine = AshareBacktestEngine()
            df_bt = engine.run(df_res)
            
            # 绘制图表
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            
            # K线与Regime颜色
            colors = ['green', 'orange', 'red'] # 0:绿(吸筹), 1:黄(震荡), 2:红(风险)
            for i in range(3):
                mask = df_res['Regime'] == i
                fig.add_trace(go.Scatter(
                    x=df_res.index[mask], y=df_res['Close'][mask],
                    mode='markers', marker=dict(color=colors[i], size=3),
                    name=f"Regime {i}"
                ), row=1, col=1)
                
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Close'], line=dict(color='gray', width=1, opacity=0.5), showlegend=False), row=1, col=1)
            
            # 资金曲线
            fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Equity_Curve'], name="策略净值", line=dict(color='red', width=2)), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Benchmark_Curve'], name="基准净值", line=dict(color='gray', dash='dot')), row=2, col=1)
            
            fig.update_layout(template="plotly_dark", height=600, title="价格体制识别与回测净值")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("未找到该股票数据，请检查代码。")

elif mode == "🛡️ 鲁棒性测试 (Robustness)":
    st.header("🛡️ A股策略参数高原测试")
    st.info("测试 HMM 自适应策略在 A 股不同参数下的稳健性。")
    
    ticker_rob = st.sidebar.text_input("测试标的", value="600519.SS")
    
    if st.sidebar.button("启动压力测试"):
        df = yf.download(ticker_rob, start=datetime.now()-timedelta(days=365*3), end=datetime.now(), progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        if not df.empty:
            # 定义扫描参数
            windows = range(15, 60, 5) # 波动率窗口
            thresholds = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006] # 开仓阈值
            
            # 调用 robustness.py 中的工具
            res_df, fig = RobustnessLab.run_sweep(
                df,
                HMMAdaptiveAshare, # 传入适配了A股的策略类
                AshareBacktestEngine, # 传入适配了A股的回测引擎
                windows,
                thresholds,
                progress_callback=st.progress(0).progress
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            avg, cv, assess = RobustnessLab.check_stability(res_df)
            st.markdown(assess)
        else:
            st.error("数据获取失败。")
