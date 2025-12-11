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

# A股核心板块 (用于扫描) - 扩充版
SECTORS = {
    "茅指数 (核心资产)": [
        "600519.SS",  # 贵州茅台 - 白酒绝对龙头
        "000858.SZ",  # 五粮液 - 浓香龙头
        "600036.SS",  # 招商银行 - 零售银行标杆
        "601318.SS",  # 中国平安 - 保险龙头
        "600276.SS",  # 恒瑞医药 - 创新药龙头
        "000333.SZ",  # 美的集团 - 家电龙头
        "300760.SZ",  # 迈瑞医疗 - 医疗器械龙头
        "603288.SS",  # 海天味业 - 调味品龙头
        "600887.SS",  # 伊利股份 - 乳制品龙头
        "000651.SZ"   # 格力电器 - 空调龙头
    ],
    
    "宁组合 (新能源)": [
        "300750.SZ",  # 宁德时代 - 动力电池全球龙头
        "002594.SZ",  # 比亚迪 - 新能源整车龙头
        "601012.SS",  # 隆基绿能 - 光伏硅片龙头
        "300274.SZ",  # 阳光电源 - 光伏逆变器龙头
        "002129.SZ",  # TCL中环 - 硅片技术领先
        "002459.SZ",  # 晶澳科技 - 光伏组件龙头
        "002466.SZ",  # 天齐锂业 - 锂矿资源龙头
        "002460.SZ",  # 赣锋锂业 - 锂资源龙头
        "300014.SZ",  # 亿纬锂能 - 锂电池第二梯队
        "002812.SZ"   # 恩捷股份 - 隔膜龙头
    ],
    
    "中特估 (红利价值)": [
        "600900.SS",  # 长江电力 - 水电龙头，高股息
        "601088.SS",  # 中国神华 - 煤炭龙头，高分红
        "600941.SS",  # 中国移动 - 电信运营商，数字经济
        "601857.SS",  # 中国石油 - 能源巨头，改革受益
        "601225.SS",  # 陕西煤业 - 优质煤企，高盈利
        "601668.SS",  # 中国建筑 - 建筑龙头，低估值
        "601390.SS",  # 中国中铁 - 基建龙头，一带一路
        "601766.SS",  # 中国中车 - 轨交装备，高端制造
        "600028.SS",  # 中国石化 - 化工龙头，炼化一体化
        "601985.SS"   # 中国核电 - 核电运营，清洁能源
    ],
    
    "AI算力 (TMT)": [
        "301308.SZ",  # 江波龙 - 存储芯片，AI算力基础设施
        "688041.SS",  # 海光信息 - AI算力芯片，国产替代 (修正后缀为.SS)
        "603019.SS",  # 中科曙光 - AI服务器龙头
        "000977.SZ",  # 浪潮信息 - 服务器龙头
        "300308.SZ",  # 中际旭创 - 光模块全球龙头
        "002230.SZ",  # 科大讯飞 - AI+教育办公应用
        "002415.SZ",  # 海康威视 - AI+安防，机器视觉
        "688256.SS",  # 寒武纪 - AI芯片设计 (修正后缀为.SS)
        "688111.SS",  # 金山办公 - AI+办公软件 (修正后缀为.SS)
        "300394.SZ"   # 天孚通信 - 光器件，CPO概念
    ],
    
    "半导体 (国产替代)": [
        "688981.SS",  # 中芯国际 - 晶圆代工龙头 (修正后缀为.SS)
        "688012.SS",  # 中微公司 - 刻蚀设备龙头 (修正后缀为.SS)
        "002371.SZ",  # 北方华创 - 半导体设备平台
        "603501.SS",  # 韦尔股份 - CIS设计龙头
        "688120.SS",  # 华海清科 - CMP设备龙头 (修正后缀为.SS)
        "688072.SS",  # 拓荆科技 - 薄膜沉积设备 (修正后缀为.SS)
        "688008.SS",  # 澜起科技 - 内存接口芯片 (修正后缀为.SS)
        "002049.SZ",  # 紫光国微 - 安全芯片龙头
        "603986.SS",  # 兆易创新 - 存储芯片设计
        "688396.SS"   # 华润微 - IDM模式龙头 (修正后缀为.SS)
    ],
    
    "医药健康 (创新药)": [
        "600276.SS",  # 恒瑞医药 - 创新药龙头
        "300760.SZ",  # 迈瑞医疗 - 医疗器械龙头
        "603259.SS",  # 药明康德 - CXO全球龙头
        "300015.SZ",  # 爱尔眼科 - 医疗服务龙头
        "600436.SS",  # 片仔癀 - 中药保密品种
        "000538.SZ",  # 云南白药 - 民族品牌中药
        "002821.SZ",  # 凯莱英 - CDMO龙头
        "688271.SS",  # 联影医疗 - 影像设备龙头 (修正后缀为.SS)
        "300122.SZ",  # 智飞生物 - 疫苗龙头
        "300347.SZ"   # 泰格医药 - 临床CRO龙头
    ],
    
    "大金融 (稳健配置)": [
        "601318.SS",  # 中国平安 - 保险综合金融
        "600036.SS",  # 招商银行 - 零售银行标杆
        "601166.SS",  # 兴业银行 - 同业业务领先
        "000001.SZ",  # 平安银行 - 科技零售银行
        "600030.SS",  # 中信证券 - 券商综合龙头
        "601688.SS",  # 华泰证券 - 财富管理领先
        "600837.SS",  # 海通证券 - 投行业务强
        "601988.SS",  # 中国银行 - 国有大行稳健
        "601628.SS",  # 中国人寿 - 寿险龙头
        "601601.SS"   # 中国太保 - 保险稳健增长
    ],
    
    "高端制造 (机器人)": [
        "002008.SZ",  # 大族激光 - 激光设备龙头
        "300124.SZ",  # 汇川技术 - 工业自动化龙头
        "002747.SZ",  # 埃斯顿 - 工业机器人本体
        "603305.SS",  # 旭升集团 - 一体化压铸
        "688017.SS",  # 绿的谐波 - 机器人减速器 (修正后缀为.SS)
        "300220.SZ",  # 金运激光 - 3D打印
        "002426.SZ",  # 胜利精密 - 智能制造
        "300607.SZ",  # 拓斯达 - 工业机器人集成
        "603611.SS",  # 诺力股份 - 智能物流装备
        "300024.SZ"   # 机器人 - 特种机器人
    ],
    
    "数字经济 (数据要素)": [
        "300212.SZ",  # 易华录 - 数据湖建设运营
        "002230.SZ",  # 科大讯飞 - AI+数据应用
        "300188.SZ",  # 国投智能 - 数据安全
        "300229.SZ",  # 拓尔思 - 大数据服务
        "000034.SZ",  # 神州数码 - 云服务和数字化
        "300663.SZ",  # 科蓝软件 - 金融科技
        "300075.SZ",  # 数字政通 - 政务数字化
        "300379.SZ",  # 东方通 - 中间件和数据
        "300010.SZ",  # 立思辰 - 教育信息化
        "300168.SZ"   # 万达信息 - 医疗信息化
    ],
    
    "消费电子 (智能驾驶)": [
        "002475.SZ",  # 立讯精密 - 消费电子龙头
        "002920.SZ",  # 德赛西威 - 智能座舱龙头
        "002241.SZ",  # 歌尔股份 - VR/AR设备
        "002456.SZ",  # 欧菲光 - 光学镜头
        "300496.SZ",  # 中科创达 - 智能操作系统
        "002906.SZ",  # 华阳集团 - 汽车电子
        "300433.SZ",  # 蓝思科技 - 玻璃盖板
        "002600.SZ",  # 领益智造 - 精密功能件
        "300433.SZ",  # 蓝思科技 - 消费电子外观 (重复项保留，作为权重)
        "002384.SZ"   # 东山精密 - FPC和PCB
    ]
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
