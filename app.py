# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import pydeck as pdk
import time
import math
import warnings

# ==========================================
# 1. 全局配置与状态初始化
# ==========================================
st.set_page_config(layout="wide", page_title="城市雨水管网水力模拟系统")
warnings.filterwarnings('ignore')

# 检查可选库依赖
try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# 初始化 Session State (关键：用于跨帧存储状态)
if 'selected_pipe_id' not in st.session_state:
    st.session_state['selected_pipe_id'] = None  # 存储当前选中的管道ID
if 'simulation_results' not in st.session_state:
    st.session_state['simulation_results'] = None # 存储计算结果
if 'pipe_id_list' not in st.session_state:
    st.session_state['pipe_id_list'] = None       # 存储ID列表以便快速查找索引

# ==========================================
# 2. 核心水力计算类 (保持原样，未修改)
# ==========================================
class VectorizedHydraulics:
    def solve_normal_depth(self, Q_target, D, S, n):
        # 防止除零和负坡度
        S = np.where(S <= 1e-6, 1e-6, S)
        sqrt_S = np.sqrt(S)
        
        # 满管流量计算 (Manning公式)
        Q_full_capacity = (1/n) * (np.pi*(D/2)**2) * ((D/4)**(2/3)) * sqrt_S
        
        # 标记超载管道
        overloaded = Q_target >= Q_full_capacity
        
        # 目标 K 值
        K_target = (Q_target * n) / sqrt_S
        
        # 初始化 theta (充满度角)
        theta = np.full_like(Q_target, np.pi, dtype=np.float64)
        
        # 筛选需要迭代求解的管道
        mask_solve = (~overloaded) & (Q_target > 0.0001)
        
        # 如果没有需要求解的，直接返回
        if not np.any(mask_solve):
             h = np.zeros_like(Q_target)
             h[overloaded] = D[overloaded]
             v = np.zeros_like(Q_target)
             full_area = np.pi * (D/2)**2
             v[overloaded] = Q_target[overloaded] / full_area[overloaded]
             return h, v

        # 提取活跃数据进行牛顿迭代
        theta_active = theta[mask_solve]
        D_active = D[mask_solve]
        K_t_active = K_target[mask_solve]
        coef_active = (D_active**2) / 8
        
        for _ in range(8): # 8次迭代通常足够收敛
            sin_t = np.sin(theta_active)
            cos_t = np.cos(theta_active)
            A = coef_active * (theta_active - sin_t)
            P = (D_active / 2) * theta_active
            P[P < 1e-6] = 1e-6
            R = A / P
            
            # 曼宁公式变形 f(theta) = A * R^(2/3) - K_target
            f_val = A * (R**(2/3)) - K_t_active
            
            # 导数计算
            dA_dth = coef_active * (1 - cos_t)
            dP_dth = D_active / 2
            term1 = (5/3) * (A**(2/3)) * (P**(-2/3)) * dA_dth
            term2 = (2/3) * (A**(5/3)) * (P**(-5/3)) * dP_dth
            f_prime = term1 - term2
            
            # 防止导数为0
            f_prime[np.abs(f_prime) < 1e-6] = 1e-6
            
            # 更新 theta
            theta_active -= f_val / f_prime
            theta_active = np.clip(theta_active, 1e-4, 2*np.pi - 1e-4)

        # 填回结果
        theta[mask_solve] = theta_active
        theta[overloaded] = 2 * np.pi
        theta[Q_target <= 0.0001] = 0
        
        # 计算水深 h 和 流速 v
        h = (D / 2) * (1 - np.cos(theta / 2))
        A_final = (D**2 / 8) * (theta - np.sin(theta))
        v = np.zeros_like(Q_target)
        valid_A = A_final > 1e-6
        v[valid_A] = Q_target[valid_A] / A_final[valid_A]
        
        return h, v

# ==========================================
# 3. 数据处理与辅助函数
# ==========================================
@st.cache_data
def load_data(file):
    """加载并标准化数据，确保ID为字符串类型"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        return None, f"文件读取错误: {e}"

    # 标准化列名
    col_map = {
        'PipeID': 'PipeID', 'ID': 'PipeID', 'name': 'PipeID',
        'UpstreamNode': 'UpstreamNode', 'US': 'UpstreamNode', 'start': 'UpstreamNode',
        'DownstreamNode': 'DownstreamNode', 'DS': 'DownstreamNode', 'end': 'DownstreamNode',
        'Diameter': 'Diameter', 'D': 'Diameter',
        'Length': 'Length', 'L': 'Length',
        'Slope': 'Slope',
        'Manning': 'Manning', 'n': 'Manning',
        'US_X': 'US_X', 'start_x': 'US_X',
        'US_Y': 'US_Y', 'start_y': 'US_Y',
        'DS_X': 'DS_X', 'end_x': 'DS_X',
        'DS_Y': 'DS_Y', 'end_y': 'DS_Y'
    }
    df = df.rename(columns=col_map)
    
    # 必填字段检查
    required = ['PipeID', 'UpstreamNode', 'DownstreamNode', 'Diameter', 'Length', 'Slope']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, f"缺失必要列: {missing}"

    # 关键修复：强制ID转换为字符串，避免 int vs str 匹配失败
    df['PipeID'] = df['PipeID'].astype(str)
    df['UpstreamNode'] = df['UpstreamNode'].astype(str)
    df['DownstreamNode'] = df['DownstreamNode'].astype(str)
    
    # 填充默认曼宁系数
    if 'Manning' not in df.columns:
        df['Manning'] = 0.013

    # 检查是否有坐标
    has_coords = all(c in df.columns for c in ['US_X', 'US_Y', 'DS_X', 'DS_Y'])
    
    return df, None, has_coords

def process_coordinates(df):
    """处理坐标转换 (HK80 -> WGS84)"""
    if not HAS_PYPROJ:
        return df, False
    
    # 简单判断是否需要转换 (假设HK80坐标值很大)
    if df['US_X'].mean() > 1000: 
        try:
            transformer = Transformer.from_crs("EPSG:2326", "EPSG:4326", always_xy=True)
            df['US_Lon'], df['US_Lat'] = transformer.transform(df['US_X'].values, df['US_Y'].values)
            df['DS_Lon'], df['DS_Lat'] = transformer.transform(df['DS_X'].values, df['DS_Y'].values)
            return df, True
        except:
            return df, False
    else:
        # 假设已经是经纬度
        df['US_Lon'], df['US_Lat'] = df['US_X'], df['US_Y']
        df['DS_Lon'], df['DS_Lat'] = df['DS_X'], df['DS_Y']
        return df, True

def build_topology(df):
    """构建有向无环图"""
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['UpstreamNode'], row['DownstreamNode'], 
                   pipe_id=row['PipeID'], length=row['Length'])
    
    # 破环处理
    cycles = 0
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G)
            G.remove_edge(*cycle[0])
            cycles += 1
        except:
            break
    return G, cycles

# ==========================================
# 4. 模拟运行逻辑
# ==========================================
def run_simulation(G, df, hours):
    """执行水力模拟"""
    solver = VectorizedHydraulics()
    
    # 拓扑排序
    try:
        topo_nodes = list(nx.topological_sort(G))
    except:
        return None, "拓扑排序失败，网络中仍有环路"

    # 生成模拟降雨/入流数据
    all_nodes = list(G.nodes())
    t_steps = np.arange(hours)
    node_inflows = {}
    for node in all_nodes:
        # 简单的随机入流模式
        base = np.random.uniform(0.001, 0.005)
        pattern = np.maximum(0.1, np.sin(t_steps/24 * 2 * np.pi) + 1)
        node_inflows[node] = base * pattern

    # 准备向量化计算所需数组
    pipe_ids = df['PipeID'].values
    pipe_map = {pid: i for i, pid in enumerate(pipe_ids)}
    num_pipes = len(df)
    
    D = df['Diameter'].values
    S = df['Slope'].values
    n = df['Manning'].values
    
    # 结果矩阵
    res_Q = np.zeros((num_pipes, hours))
    res_v = np.zeros((num_pipes, hours))
    res_h = np.zeros((num_pipes, hours))
    
    # 逐小时计算
    progress = st.progress(0)
    for t in range(hours):
        progress.progress((t+1)/hours)
        
        # 当前时刻节点累积流量
        current_node_flow = {node: node_inflows[node][t] for node in all_nodes}
        current_pipe_Q = np.zeros(num_pipes)
        
        # 流量传导
        for u in topo_nodes:
            inflow = current_node_flow[u]
            out_edges = G.out_edges(u, data=True)
            if not out_edges: continue
            
            # 简单平均分配到下游
            q_per_pipe = inflow / len(out_edges)
            for _, v, data in out_edges:
                pid = data['pipe_id']
                if pid in pipe_map:
                    idx = pipe_map[pid]
                    current_pipe_Q[idx] = q_per_pipe
                    current_node_flow[v] += q_per_pipe
        
        # 水力解算
        h_t, v_t = solver.solve_normal_depth(current_pipe_Q, D, S, n)
        
        res_Q[:, t] = current_pipe_Q
        res_v[:, t] = v_t
        res_h[:, t] = h_t
        
    progress.empty()
    
    return {
        'Q': res_Q, 'v': res_v, 'h': res_h, 'time': t_steps
    }, None

# ==========================================
# 5. 界面主逻辑
# ==========================================
st.title("🌊 城市雨水管网水力分析系统")

# --- 侧边栏：输入与控制 ---
with st.sidebar:
    st.header("1. 数据配置")
    uploaded_file = st.file_uploader("上传管网数据 (Excel/CSV)", type=['xlsx', 'csv'])
    
    st.header("2. 模拟参数")
    sim_hours = st.slider("模拟时长 (h)", 12, 48, 24)
    
    run_btn = st.button("🚀 运行模拟", type="primary", use_container_width=True)

# --- 数据加载与处理 ---
if uploaded_file:
    df_raw, err, has_coords = load_data(uploaded_file)
    if err:
        st.error(err)
        st.stop()
    
    # 坐标转换
    df_map = df_raw.copy()
    if has_coords:
        df_map, success = process_coordinates(df_map)
        if not success:
            st.warning("坐标转换失败，地图可能无法正确显示")

    # --- 模拟触发 ---
    if run_btn:
        G, cycles = build_topology(df_raw)
        if cycles > 0:
            st.warning(f"自动处理了 {cycles} 处环路")
        
        results, msg = run_simulation(G, df_raw, sim_hours)
        if results:
            st.session_state['simulation_results'] = results
            st.session_state['pipe_id_list'] = df_raw['PipeID'].values # 保存ID顺序以便索引
            st.success("✅ 模拟完成！请在右侧点击管道查看结果。")
            st.rerun() # 刷新以更新状态
        else:
            st.error(msg)

    # --- 主界面布局：左图右表 ---
    col_map, col_info = st.columns([2, 1])

    # === 左侧：地图交互 (核心修改部分) ===
    with col_map:
        if has_coords and 'US_Lon' in df_map.columns:
            # 1. 准备地图数据与样式
            # 这里的逻辑是：给 df_map 增加颜色列，根据 session_state 中的选中ID 动态改变颜色
            current_id = st.session_state['selected_pipe_id']
            
            def get_color(pid):
                if str(pid) == str(current_id):
                    return [255, 0, 255, 255] # 选中：品红
                return [0, 191, 255, 200]     # 默认：深天蓝
            
            def get_width(pid):
                if str(pid) == str(current_id):
                    return 8
                return 3

            df_map['color'] = df_map['PipeID'].apply(get_color)
            df_map['width'] = df_map['PipeID'].apply(get_width)

            # 2. 定义图层
            # 管道层 (LineLayer) - 开启 Pickable
            layer_pipes = pdk.Layer(
                "LineLayer",
                df_map,
                get_source_position=['US_Lon', 'US_Lat'],
                get_target_position=['DS_Lon', 'DS_Lat'],
                get_color='color',
                get_width='width',
                pickable=True,  # 允许点击
                auto_highlight=True,
            )

            # 节点层 (ScatterplotLayer) - 关闭 Pickable
            # 避免点击节点时返回错误的索引，干扰管道选择
            nodes_data = df_map[['US_Lon', 'US_Lat']].drop_duplicates()
            layer_nodes = pdk.Layer(
                "ScatterplotLayer",
                nodes_data,
                get_position=['US_Lon', 'US_Lat'],
                get_radius=15,
                get_fill_color=[255, 255, 255, 150],
                pickable=False, # 禁止点击
            )

            # 3. 视图设置
            view_state = pdk.ViewState(
                latitude=df_map['US_Lat'].mean(),
                longitude=df_map['US_Lon'].mean(),
                zoom=14,
                pitch=0
            )

            # 4. 渲染地图并捕获事件
            deck = pdk.Deck(
                layers=[layer_pipes, layer_nodes],
                initial_view_state=view_state,
                tooltip={"text": "管道ID: {PipeID}\n管径: {Diameter}m"}
            )
            
            # 使用 on_select="rerun" 确保点击后立即刷新 Streamlit
            event = st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object", use_container_width=True, height=600)

            # 5. 处理点击事件 (修复Bug的核心)
            if event.selection:
                indices = event.selection.get('indices')
                if indices:
                    # 获取被点击行的索引
                    idx = indices[0]
                    # 从数据源中找到对应的 PipeID
                    # 注意：PyDeck 的索引对应的是传入 Layer 的 DataFrame 的行号
                    if idx < len(df_map):
                        clicked_id = str(df_map.iloc[idx]['PipeID'])
                        
                        # 如果点击了不同的管道，更新状态并刷新
                        if clicked_id != st.session_state['selected_pipe_id']:
                            st.session_state['selected_pipe_id'] = clicked_id
                            st.rerun() # 强制刷新，使右侧面板和地图颜色更新
        else:
            st.info("数据中未检测到有效坐标，无法显示地图。")

    # === 右侧：结果展示 ===
    with col_info:
        selected_id = st.session_state['selected_pipe_id']
        results = st.session_state['simulation_results']
        
        st.subheader("📊 模拟结果详情")
        
        if not selected_id:
            st.info("👈 请在左侧地图点击管道查看详情")
        elif not results:
            st.warning(f"已选中管道: **{selected_id}**\n\n⚠️ 暂无模拟结果，请先点击左侧侧边栏的“运行模拟”按钮。")
        else:
            # 查找结果索引
            try:
                # 确保类型匹配 (str vs str)
                id_list = st.session_state['pipe_id_list'].astype(str)
                # np.where 返回的是 tuple, 取第一个元素
                idx_arr = np.where(id_list == str(selected_id))[0]
                
                if len(idx_arr) == 0:
                    st.error(f"在结果集中未找到管道 ID: {selected_id}")
                else:
                    idx = idx_arr[0]
                    
                    # 提取数据
                    ts_Q = results['Q'][idx]
                    ts_v = results['v'][idx]
                    ts_h = results['h'][idx]
                    times = results['time']
                    
                    # 基础信息卡片
                    pipe_info = df_raw[df_raw['PipeID'].astype(str) == str(selected_id)].iloc[0]
                    st.markdown(f"**管道 ID:** `{selected_id}`")
                    c1, c2 = st.columns(2)
                    c1.metric("管径 (m)", f"{pipe_info['Diameter']:.3f}")
                    c2.metric("管长 (m)", f"{pipe_info['Length']:.1f}")
                    
                    st.divider()
                    
                    # 绘图
                    if HAS_PLOTLY:
                        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                          vertical_spacing=0.1,
                                          subplot_titles=("流量 Q (m³/s)", "流速 v (m/s)", "水深 h (m)"))
                        
                        fig.add_trace(go.Scatter(x=times, y=ts_Q, name="流量", fill='tozeroy'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=times, y=ts_v, name="流速"), row=2, col=1)
                        fig.add_trace(go.Scatter(x=times, y=ts_h, name="水深", fill='tozeroy'), row=3, col=1)
                        
                        # 添加管顶警戒线
                        fig.add_hline(y=pipe_info['Diameter'], line_dash="dash", line_color="red", row=3, col=1)
                        
                        fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.line_chart(pd.DataFrame({'流量': ts_Q, '流速': ts_v, '水深': ts_h}))
                        
            except Exception as e:
                st.error(f"数据查询出错: {str(e)}")

else:
    st.info("👈 请先上传数据文件开始。")
