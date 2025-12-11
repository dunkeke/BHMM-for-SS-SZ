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
# PART 2: AI 智能投顾模块 (强化版)
# ==========================================

class AI_Investment_Advisor:
    """
    AI 投资顾问: 将量化数据翻译为人类可读的投资建议
    """
    @staticmethod
    def analyze(df, metrics, strategy_type):
        last = df.iloc[-1]
        # 获取前一天的信号，用于判断突变
        prev = df.iloc[-2] if len(df) > 1 else last
        
        regime = int(last['Regime'])
        signal = int(last['Signal'])
        prev_signal = int(prev['Signal'])
        alpha = last.get('Bayes_Exp_Ret', 0)
        
        # 1. 判断信号突变 (Signal Flip)
        signal_change = "None"
        if signal == 1 and prev_signal == 0:
            signal_change = "BUY_NEW" # 新增买点
        elif signal == 0 and prev_signal == 1:
            signal_change = "SELL_EXIT" # 新增卖点
        
        # 2. 市场状态画像
        regime_desc = {
            0: "🌱 底部/吸筹 (Low Vol)",
            1: "🌊 趋势/中继 (Med Vol)", 
            2: "🌪️ 顶部/风险 (High Vol)"
        }
        market_status = regime_desc.get(regime, "未知状态")
        
        # 3. 策略逻辑解释
        logic_expl = ""
        if strategy_type == 'Standard':
            logic_expl = f"HMM 处于 {market_status}。"
        elif strategy_type == 'Adaptive':
            logic_expl = f"贝叶斯 Alpha={alpha*100:.3f}% ({'积极' if alpha>0 else '消极'})。"
        elif strategy_type == 'MACD_Resonance':
            macd_val = last.get('MACD_Hist', 0)
            logic_expl = f"HMM {'看多' if last.get('HMM_Signal',0)==1 else '看空'} + MACD {'金叉' if macd_val>0 else '死叉'}。"

        # 4. 最终行动建议 (结合突变判断)
        advice_card = {
            "action_title": "",
            "action_color": "",
            "bg_color": "",
            "summary": "",
            "risk_warning": "",
            "signal_change": signal_change # 传递突变状态
        }
        
        if signal == 1:
            if signal_change == "BUY_NEW":
                advice_card['action_title'] = "🔔 信号突变：买入建仓 (BUY ALERT)"
                advice_card['summary'] = f"**{market_status}**。今日策略信号由空转多！{logic_expl} 建议把握建仓时机。"
            else:
                advice_card['action_title'] = "🚀 强力持股 (HOLD)"
                advice_card['summary'] = f"**{market_status}**。多头趋势延续中。{logic_expl} 建议坚定持有。"
            
            advice_card['action_color'] = "#00E676" # Green
            advice_card['bg_color'] = "rgba(0, 230, 118, 0.1)"
            advice_card['risk_warning'] = "止损建议：若跌破20日均线或HMM跳变至State 2，立即离场。"
            
        else:
            if signal_change == "SELL_EXIT":
                advice_card['action_title'] = "🔔 信号突变：离场警报 (EXIT ALERT)"
                advice_card['summary'] = f"**{market_status}**。今日策略信号由多转空！{logic_expl} 风险显著增加，建议立即卖出。"
            else:
                advice_card['action_title'] = "🛡️ 空仓观望 (WAIT)"
                advice_card['summary'] = f"**{market_status}**。当前无操作机会。{logic_expl} 建议持有现金，等待新信号。"
                
            advice_card['action_color'] = "#FF5252" # Red
            advice_card['bg_color'] = "rgba(255, 82, 82, 0.1)"
            advice_card['risk_warning'] = "观察建议：耐心等待 HMM 状态回归 State 0。"
            
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
                prev = df.iloc[-2]
                
                # 信号突变检测
                change = "不变"
                if last['Signal']==1 and prev['Signal']==0: change = "🚀 新买点"
                elif last['Signal']==0 and prev['Signal']==1: change = "🔻 离场"
                
                # 评分
                score = last.get('Bayes_Exp_Ret', 0) * 10000
                if 'MACD_Hist' in df.columns: score += last['MACD_Hist'] * 100 
                
                results.append({
                    "代码": ticker,
                    "最新价": last['Close'],
                    "HMM状态": int(last['Regime']),
                    "当前信号": "🟢 持股" if last['Signal']==1 else "⚪ 空仓",
                    "异动提醒": change,
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
                
                # 3. 生成 AI 建议
                advice = AI_Investment_Advisor.analyze(df_res, metrics, df_res['Strategy_Type'].iloc[-1])
                
                # --- UI 展示 ---
                
                # A. 信号突变横幅 (Alert Banner)
                if advice['signal_change'] == "BUY_NEW":
                    st.success("🚨 **ALERT: DETECTED NEW BUY SIGNAL TODAY (今日触发买入信号)**")
                elif advice['signal_change'] == "SELL_EXIT":
                    st.error("🚨 **ALERT: DETECTED EXIT SIGNAL TODAY (今日触发卖出信号)**")

                # B. AI 建议卡片
                st.markdown(f"""
                <div style="background:{advice['bg_color']}; padding:20px; border-radius:12px; border-left:6px solid {advice['action_color']}; margin-bottom:20px;">
                    <h2 style="color:{advice['action_color']}; margin:0;">{advice['action_title']}</h2>
                    <p style="color:#EEE; font-size:1.1em; margin-top:10px;">{advice['summary']}</p>
                    <hr style="border-color:rgba(255,255,255,0.1);">
                    <p style="color:#AAA; font-size:0.9em;">⚠️ <strong>风控提示</strong>: {advice['risk_warning']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # C. 核心指标
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("策略总回报", f"{metrics['Total Return']*100:.1f}%")
                k2.metric("夏普比率", f"{metrics['Sharpe']:.2f}")
                k3.metric("最大回撤", f"{metrics['Max Drawdown']*100:.1f}%")
                k4.metric("当前 Alpha (bps)", f"{df_res['Bayes_Exp_Ret'].iloc[-1]*10000:.1f}")
                
                # D. 图表 (增加买卖点标记)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.4])
                
                # K线 & 状态背景点
                colors = ['#00E676', '#FFD600', '#FF1744'] 
                for i in range(3):
                    mask = df_res['Regime'] == i
                    fig.add_trace(go.Scatter(x=df_res.index[mask], y=df_res['Close'][mask], mode='markers', marker=dict(color=colors[i], size=3), name=f"Regime {i}"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Close'], line=dict(color='gray', width=1), opacity=0.5, showlegend=False), row=1, col=1)
                
                # *** 新增：明确的买卖点箭头 ***
                # 计算信号变化点: 0->1 (Buy), 1->0 (Sell)
                df_res['Signal_Diff'] = df_res['Signal'].diff()
                buy_points = df_res[df_res['Signal_Diff'] == 1]
                sell_points = df_res[df_res['Signal_Diff'] == -1]
                
                if not buy_points.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_points.index, y=buy_points['Close']*0.98, # 稍微在K线下放一点
                        mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00E676'),
                        name='明确买点 (Buy Action)'
                    ), row=1, col=1)
                    
                if not sell_points.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_points.index, y=sell_points['Close']*1.02, # 稍微在K线上放一点
                        mode='markers', marker=dict(symbol='triangle-down', size=12, color='#FF5252'),
                        name='明确卖点 (Sell Action)'
                    ), row=1, col=1)

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
                
                # 推荐展示 (异动优先)
                new_actions = res_df[res_df['异动提醒'].isin(["🚀 新买点", "🔻 离场"])]
                if not new_actions.empty:
                    st.info(f"⚡ **今日异动 (Signal Change Today):** {len(new_actions)} 只标的触发信号突变！")
                    st.dataframe(new_actions, use_container_width=True, hide_index=True)
                
                # 现有持仓推荐
                top_buys = res_df[res_df['当前信号'].str.contains("持股")]
                if not top_buys.empty:
                    st.success(f"🎯 **持股池 (Holding):** {len(top_buys)} 只标的建议继续持有")
                    with st.expander("查看持股列表"):
                        st.dataframe(top_buys, use_container_width=True, hide_index=True)
                else:
                    st.warning("当前板块无持股建议，建议观望。")
                
                with st.expander("查看完整扫描结果"):
                    st.dataframe(res_df, use_container_width=True)
            else:
                st.error("数据获取失败。")
