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
    pass # 鲁棒性模块为可选

warnings.filterwarnings("ignore")

# ==========================================
# 0. 全局配置
# ==========================================
st.set_page_config(page_title="A-Share AI Quant Pro", layout="wide", page_icon="🇨🇳")

# A股核心板块 (用于扫描)
SECTORS = {
    "茅指数 (核心资产)": ["600519.SS", "000858.SZ", "600036.SS", "601318.SS", "600276.SS", "000333.SZ"],
    "宁组合 (新能源)": ["300750.SZ", "002594.SZ", "601012.SS", "300274.SZ", "002475.SZ", "002371.SZ"],
    "中特估 (红利)": ["600900.SS", "601088.SS", "600941.SS", "601288.SS", "601225.SS", "601006.SS"],
    "AI算力 (TMT)": ["601138.SS", "002230.SZ", "603019.SS", "000977.SZ", "300308.SZ", "002920.SZ"]
}

# A股交易成本 (双边万5 + 滑点)
ASHARE_COST = 0.0005

# ==========================================
# PART 1: 策略工厂 (Strategy Zoo for A-Share)
# ==========================================

class StrategyBase:
    """策略基类"""
    def generate_signals(self, df): raise NotImplementedError

class HMMStandardAshare(StrategyBase):
    """
    [经典策略 - A股版]
    逻辑: 低波(State 0) -> 买入, 高波(State 2) -> 卖出/空仓
    适配: 只能做多 (Signal >= 0)
    """
    def __init__(self, n_components=3, iter_num=1000, window_size=21, **kwargs):
        self.n_components = n_components
        self.iter_num = iter_num
        self.window_size = window_size

    def generate_signals(self, df):
        df = df.copy()
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Log_Ret'].rolling(window=self.window_size).std()
        df.dropna(inplace=True)
        if len(df) < 60: return df
        
        X = df[['Log_Ret', 'Volatility']].values * 100.0
        try:
            model = GaussianHMM(n_components=self.n_components, covariance_type="full", n_iter=self.iter_num, random_state=42, tol=0.01, min_covar=0.01)
            model.fit(X)
        except: return df
        
        hidden_states = model.predict(X)
        # 按波动率排序状态
        state_vol_means = [X[hidden_states == i, 1].mean() for i in range(self.n_components)]
        sorted_stats = sorted(list(enumerate(state_vol_means)), key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
        df['Regime'] = np.array([mapping[s] for s in hidden_states])
        
        # A股逻辑: Regime 0 (低波) -> 买入; Regime 2 (高波) -> 卖出
        df['Signal'] = 0
        df.loc[df['Regime'] == 0, 'Signal'] = 1   
        # 其他状态保持 0 (空仓)
        
        # 补充字段用于 AI 分析
        df['Bayes_Exp_Ret'] = 0.0 # 标准版不计算贝叶斯
        df['Strategy_Type'] = 'Standard'
        return df

class HMMAdaptiveAshare(StrategyBase):
    """
    [自适应策略 - A股版]
    逻辑: 基于贝叶斯后验期望收益 > 阈值 -> 买入
    适配: Long Only
    """
    def __init__(self, n_components=3, iter_num=1000, window_size=21, threshold=0.0003, **kwargs):
        self.n_components = n_components
        self.iter_num = iter_num
        self.window_size = window_size
        self.threshold = threshold

    def generate_signals(self, df):
        df = df.copy()
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Log_Ret'].rolling(window=self.window_size).std()
        df.dropna(inplace=True)
        if len(df) < 60: return df
        
        X = df[['Log_Ret', 'Volatility']].values * 100.0
        try:
            model = GaussianHMM(n_components=self.n_components, covariance_type="full", n_iter=self.iter_num, random_state=88, tol=0.01, min_covar=0.01)
            model.fit(X)
        except: return df
        
        hidden_states = model.predict(X)
        state_vol_means = [X[hidden_states == i, 1].mean() for i in range(self.n_components)]
        sorted_stats = sorted(list(enumerate(state_vol_means)), key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
        
        posterior_probs = model.predict_proba(X)
        sorted_probs = np.zeros_like(posterior_probs)
        for old_i, new_i in mapping.items():
            sorted_probs[:, new_i] = posterior_probs[:, old_i]
            
        df['Regime'] = np.array([mapping[s] for s in hidden_states])
        
        state_means = []
        for i in range(self.n_components):
            mean_ret = df[df['Regime'] == i]['Log_Ret'].mean()
            state_means.append(mean_ret)
        
        new_transmat = np.zeros_like(model.transmat_)
        for i in range(self.n_components):
            for j in range(self.n_components):
                new_transmat[mapping[i], mapping[j]] = model.transmat_[i, j]
                
        next_probs = np.dot(sorted_probs, new_transmat)
        df['Bayes_Exp_Ret'] = np.dot(next_probs, state_means)
        
        # A股逻辑: Alpha > 阈值 -> 买入; 否则空仓
        df['Signal'] = 0
        df.loc[df['Bayes_Exp_Ret'] > self.threshold, 'Signal'] = 1
        
        df['Strategy_Type'] = 'Adaptive'
        return df

class HMM_MACD_Ashare(StrategyBase):
    """
    [MACD共振策略 - A股版]
    逻辑: HMM 看多 + MACD 金叉 -> 买入
    """
    def __init__(self, n_components=3, iter_num=1000, window_size=21, **kwargs):
        self.n_components = n_components
        self.iter_num = iter_num
        self.window_size = window_size

    def calculate_macd(self, df):
        # 使用日线 MACD (为了简化计算，不请求4H数据，直接用日线)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        dif = exp1 - exp2
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = (dif - dea) * 2
        return hist, dif

    def generate_signals(self, df):
        # 1. 先跑基础 HMM 自适应
        base_strat = HMMAdaptiveAshare(self.n_components, self.iter_num, self.window_size)
        df = base_strat.generate_signals(df)
        if 'Signal' not in df.columns: return df
        
        # 2. 计算 MACD
        hist, dif = self.calculate_macd(df)
        df['MACD_Hist'] = hist
        df['MACD_DIF'] = dif
        
        # 3. 共振逻辑
        # 原始 HMM 信号为 1 (看多) 且 MACD 红柱扩大或为正 -> 买入
        # 如果 HMM 看多 但 MACD 死叉 -> 观望 (Signal=0)
        
        df['HMM_Signal'] = df['Signal'] # 备份 HMM 信号
        df['Signal'] = 0 # 重置
        
        # 买入条件: HMM看多 且 (MACD柱子 > 0)
        buy_condition = (df['HMM_Signal'] == 1) & (df['MACD_Hist'] > 0)
        df.loc[buy_condition, 'Signal'] = 1
        
        df['Strategy_Type'] = 'MACD_Resonance'
        return df

# ==========================================
# PART 2: AI 智能投顾模块 (核心新增)
# ==========================================

class AI_Investment_Advisor:
    """
    AI 投资顾问: 将量化数据翻译为人类可读的投资建议
    """
    @staticmethod
    def analyze(df, metrics, strategy_type):
        last = df.iloc[-1]
        regime = int(last['Regime'])
        signal = int(last['Signal'])
        alpha = last.get('Bayes_Exp_Ret', 0)
        
        # 1. 市场状态画像
        regime_desc = {
            0: "🌱 底部/吸筹 (Low Volatility)",
            1: "🌊 趋势/中继 (Medium Volatility)", 
            2: "🌪️ 顶部/风险 (High Volatility)"
        }
        market_status = regime_desc.get(regime, "未知状态")
        
        # 2. 策略逻辑解释
        logic_expl = ""
        if strategy_type == 'Standard':
            logic_expl = "经典轮动逻辑：当前处于" + ("低波稳态，符合买入条件。" if regime==0 else "高波/震荡态，建议空仓防御。")
        elif strategy_type == 'Adaptive':
            logic_expl = f"贝叶斯概率逻辑：模型预测次日具有 {'正向' if alpha>0 else '负向'} 预期收益 (Alpha={alpha*100:.3f}%)，" + ("资金做多意愿强。" if signal==1 else "风险溢价不足，建议观望。")
        elif strategy_type == 'MACD_Resonance':
            macd_val = last.get('MACD_Hist', 0)
            logic_expl = f"趋势共振逻辑：HMM 宏观判断{'看多' if last.get('HMM_Signal',0)==1 else '看空'}，叠加 MACD 技术面{'金叉(红柱)' if macd_val>0 else '死叉(绿柱)'}。" + ("双重验证通过，强烈看多。" if signal==1 else "共振失败，保持防守。")

        # 3. 最终行动建议
        advice_card = {
            "action_title": "",
            "action_color": "",
            "bg_color": "",
            "summary": "",
            "risk_warning": ""
        }
        
        if signal == 1:
            advice_card['action_title'] = "🚀 强力买入 / 持股 (LONG)"
            advice_card['action_color'] = "#00E676" # Green
            advice_card['bg_color'] = "rgba(0, 230, 118, 0.1)"
            advice_card['summary'] = f"**{market_status}**。{logic_expl} 量化信号积极，建议建立多头仓位。"
            advice_card['risk_warning'] = "止损建议：若收盘价跌破20日均线，或HMM状态跳变为State 2，立即离场。"
        else:
            advice_card['action_title'] = "🛡️ 空仓观望 / 卖出 (CASH)"
            advice_card['action_color'] = "#FF5252" # Red
            advice_card['bg_color'] = "rgba(255, 82, 82, 0.1)"
            advice_card['summary'] = f"**{market_status}**。{logic_expl} 量化信号转弱或风险过高，建议持有现金。"
            advice_card['risk_warning'] = "观察建议：等待HMM状态回归State 0，或预期Alpha转正后再行介入。"
            
        return advice_card

# ==========================================
# PART 3: 扫描与回测引擎
# ==========================================

class AshareBacktestEngine:
    """A股专用回测 (T+1, 无做空)"""
    def __init__(self, initial_capital=100000, cost=ASHARE_COST):
        self.initial_capital = initial_capital
        self.cost = cost

    def run(self, df):
        df = df.copy()
        df['Position'] = df['Signal'].shift(1).fillna(0) # T+1
        trades = df['Position'].diff().abs().fillna(0)
        fees = trades * self.cost
        df['Strategy_Ret'] = (df['Position'] * df['Log_Ret']) - fees
        df['Equity_Curve'] = self.initial_capital * (1 + df['Strategy_Ret']).cumprod()
        df['Benchmark_Curve'] = self.initial_capital * (1 + df['Log_Ret']).cumprod()
        return df
        
    def calculate_metrics(self, df):
        if df.empty: return {}
        total_ret = df['Equity_Curve'].iloc[-1]/self.initial_capital - 1
        ann_ret = (1+total_ret)**(252/len(df))-1
        vol = df['Strategy_Ret'].std()*np.sqrt(252)
        sharpe = (df['Strategy_Ret'].mean()*252)/(vol+1e-8)
        dd = (df['Equity_Curve']/df['Equity_Curve'].cummax()-1).min()
        return {"Total Return": total_ret, "CAGR": ann_ret, "Sharpe": sharpe, "Max Drawdown": dd}

def run_scanner(sector_list, strategy_cls):
    """通用扫描器"""
    results = []
    progress_bar = st.progress(0)
    for i, ticker in enumerate(sector_list):
        try:
            df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                strat = strategy_cls()
                df = strat.generate_signals(df)
                last = df.iloc[-1]
                
                # 评分
                score = last.get('Bayes_Exp_Ret', 0) * 10000
                if 'MACD_Hist' in df.columns: score += last['MACD_Hist'] * 100 # MACD加分
                
                results.append({
                    "代码": ticker,
                    "最新价": last['Close'],
                    "HMM状态": int(last['Regime']),
                    "信号": "🟢 买入" if last['Signal']==1 else "⚪ 观望",
                    "Score": score
                })
        except: pass
        progress_bar.progress((i+1)/len(sector_list))
    return pd.DataFrame(results)

# ==========================================
# PART 4: Streamlit 主程序
# ==========================================

st.title("🇨🇳 A-Share Quant Pro: AI 智能投顾")

# 侧边栏
mode = st.sidebar.radio("系统模式", ["📈 个股深度分析 (Deep Dive)", "📡 板块雷达扫描 (Scanner)"])
st.sidebar.markdown("---")
strategy_name = st.sidebar.selectbox("策略内核", ["HMM 自适应贝叶斯 (推荐)", "HMM + MACD 共振", "HMM 经典标准版"])

# 策略映射
STRAT_MAP = {
    "HMM 自适应贝叶斯 (推荐)": HMMAdaptiveAshare,
    "HMM + MACD 共振": HMM_MACD_Ashare,
    "HMM 经典标准版": HMMStandardAshare
}
CurrentStrategy = STRAT_MAP[strategy_name]

if mode == "📈 个股深度分析 (Deep Dive)":
    ticker_in = st.sidebar.text_input("A股代码 (如 600519)", value="600519")
    # 自动后缀
    full_ticker = ticker_in + (".SS" if ticker_in.startswith("6") else ".SZ") if "." not in ticker_in else ticker_in
    
    if st.sidebar.button("启动 AI 分析", type="primary"):
        with st.spinner(f"AI 正在分析 {full_ticker} 的量化特征..."):
            df = yf.download(full_ticker, period="3y", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            if not df.empty:
                # 1. 运行策略
                strat = CurrentStrategy()
                df_res = strat.generate_signals(df)
                
                # 2. 运行回测
                engine = AshareBacktestEngine()
                df_bt = engine.run(df_res)
                metrics = engine.calculate_metrics(df_bt)
                
                # 3. 生成 AI 建议 (核心功能)
                advice = AI_Investment_Advisor.analyze(df_res, metrics, df_res['Strategy_Type'].iloc[-1])
                
                # --- UI 展示 ---
                
                # A. AI 建议卡片
                st.markdown(f"""
                <div style="background:{advice['bg_color']}; padding:20px; border-radius:12px; border-left:6px solid {advice['action_color']}; margin-bottom:20px;">
                    <h2 style="color:{advice['action_color']}; margin:0;">{advice['action_title']}</h2>
                    <p style="color:#EEE; font-size:1.1em; margin-top:10px;">{advice['summary']}</p>
                    <hr style="border-color:rgba(255,255,255,0.1);">
                    <p style="color:#AAA; font-size:0.9em;">⚠️ <strong>风控提示</strong>: {advice['risk_warning']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # B. 核心指标
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("策略总回报", f"{metrics['Total Return']*100:.1f}%")
                k2.metric("夏普比率", f"{metrics['Sharpe']:.2f}")
                k3.metric("最大回撤", f"{metrics['Max Drawdown']*100:.1f}%")
                k4.metric("当前 Alpha (bps)", f"{df_res['Bayes_Exp_Ret'].iloc[-1]*10000:.1f}")
                
                # C. 图表
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.4])
                
                # K线 & 状态
                colors = ['#00E676', '#FFD600', '#FF1744'] # 绿(0), 黄(1), 红(2)
                for i in range(3):
                    mask = df_res['Regime'] == i
                    fig.add_trace(go.Scatter(x=df_res.index[mask], y=df_res['Close'][mask], mode='markers', marker=dict(color=colors[i], size=3), name=f"Regime {i}"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Close'], line=dict(color='gray', width=1), opacity=0.5, showlegend=False), row=1, col=1)
                
                # 净值
                fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Equity_Curve'], name="策略净值", line=dict(color='#2962FF', width=2)), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Benchmark_Curve'], name="基准", line=dict(color='gray', dash='dot')), row=2, col=1)
                
                fig.update_layout(template="plotly_dark", height=600, margin=dict(t=30, b=30))
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("数据获取失败，请检查代码。")

elif mode == "📡 板块雷达扫描 (Scanner)":
    sec_name = st.selectbox("选择赛道", list(SECTORS.keys()))
    if st.button("开始雷达扫描", type="primary"):
        with st.spinner(f"正在用 {strategy_name} 扫描 {sec_name}..."):
            res_df = run_scanner(SECTORS[sec_name], CurrentStrategy)
            
            if not res_df.empty:
                res_df = res_df.sort_values(by="Score", ascending=False)
                
                # 推荐展示
                top_buys = res_df[res_df['信号'].str.contains("买入")]
                if not top_buys.empty:
                    st.success(f"🎯 发现 {len(top_buys)} 只买入信号标的！")
                    st.dataframe(top_buys, use_container_width=True, hide_index=True)
                else:
                    st.warning("当前板块无买入信号，建议观望。")
                
                with st.expander("查看完整列表"):
                    st.dataframe(res_df, use_container_width=True)
            else:
                st.error("数据获取失败。")
