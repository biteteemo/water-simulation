# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import pydeck as pdk
import warnings
import time

# ==========================================
# 0. 基础配置与依赖检查
# ==========================================
st.set_page_config(
    page_title="城市雨水管网水力模拟系统",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

# 检查可选依赖
try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# 初始化 Session State
DEFAULT_STATE = {
    'selected_pipe_id': None,
    'pipe_selector': None,
    'has_results': False,
    'res_Q': None,
    'res_v': None,
    'res_h': None,
    'all_pipe_ids': None,
    'simulation_params': {} # 用于存储上次模拟的参数以判断是否需要重算
}

for key, val in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==========================================
# 1. 核心计算类 (Vectorized Hydraulics)
# ==========================================
class VectorizedHydraulics:
    """
    基于曼宁公式的矢量化水力计算器
    使用牛顿-拉夫逊迭代法求解圆形管道的正常水深
    """
    def solve_normal_depth(self, Q_target, D, S, n):
        # 避免坡度为0导致除零错误
        S = np.where(S <= 1e-6, 1e-6, S)
        sqrt_S = np.sqrt(S)
        
        # 计算满管流量 Q_full
        # Manning: Q = (1/n) * A * R^(2/3) * S^(1/2)
        # Full: A = pi*(D/2)^2, P = pi*D, R = D/4
        Q_full_capacity = (1/n) * (np.pi*(D/2)**2) * ((D/4)**(2/3)) * sqrt_S
        
        # 标记过载管道
        overloaded = Q_target >= Q_full_capacity
        
        # 准备迭代求解
        # K_target = A * R^(2/3) = (Q * n) / sqrt(S)
        K_target = (Q_target * n) / sqrt_S
        
        # 初始化 theta (充满角), 范围 [0, 2pi]
        theta = np.full_like(Q_target, np.pi, dtype=np.float64)
        
        # 只需要计算未过载且有流量的管道
        mask_solve = (~overloaded) & (Q_target > 0.0001)
        
        # 如果没有需要求解的，直接返回结果
        if not np.any(mask_solve):
             h = np.zeros_like(Q_target)
             h[overloaded] = D[overloaded] # 过载则水深为管径
             v = np.zeros_like(Q_target)
             full_area = np.pi * (D/2)**2
             v[overloaded] = Q_target[overloaded] / full_area[overloaded]
             return h, v

        # 提取需要计算的部分数据
        theta_active = theta[mask_solve]
        D_active = D[mask_solve]
        K_t_active = K_target[mask_solve]
        coef_active = (D_active**2) / 8
        
        # 牛顿迭代 (8次通常足够收敛)
        for _ in range(8):
            sin_t = np.sin(theta_active)
            cos_t = np.cos(theta_active)
            
            # A = (D^2/8) * (theta - sin(theta))
            A = coef_active * (theta_active - sin_t)
            
            # P = (D/2) * theta
            P = (D_active / 2) * theta_active
            P[P < 1e-6] = 1e-6 # 避免除零
            
            R = A / P
            
            # f(theta) = A * R^(2/3) - K_target
            f_val = A * (R**(2/3)) - K_t_active
            
            # 导数计算
            dA_dth = coef_active * (1 - cos_t)
            dP_dth = D_active / 2
            
            # f'(theta) = (5/3) * A^(2/3) * P^(-2/3) * dA - (2/3) * A^(5/3) * P^(-5/3) * dP
            term1 = (5/3) * (A**(2/3)) * (P**(-2/3)) * dA_dth
            term2 = (2/3) * (A**(5/3)) * (P**(-5/3)) * dP_dth
            f_prime = term1 - term2
            
            f_prime[np.abs(f_prime) < 1e-6] = 1e-6 # 避免除零
            
            # 更新 theta
            theta_active -= f_val / f_prime
            
            # 限制范围，防止发散
            theta_active = np.clip(theta_active, 1e-4, 2*np.pi - 1e-4)

        # 填回结果
        theta[mask_solve] = theta_active
        theta[overloaded] = 2 * np.pi
        theta[Q_target <= 0.0001] = 0
        
        # 计算最终 h 和 v
        h = (D / 2) * (1 - np.cos(theta / 2))
        A_final = (D**2 / 8) * (theta - np.sin(theta))
        
        v = np.zeros_like(Q_target)
        valid_A = A_final > 1e-6
        v[valid_A] = Q_target[valid_A] / A_final[valid_A]
        
        return h, v

# ==========================================
# 2. 数据处理函数
# ==========================================
@st.cache_data
def load_and_process_data(uploaded_file):
    """读取并预处理数据"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        return None, f"文件读取失败: {str(e)}", False

    # 标准化列名
    rename_map = {
        'name': 'PipeID', 'Pipe': 'PipeID', 'pipe_id': 'PipeID', 'ID': 'PipeID',
        'start': 'UpstreamNode', 'US': 'UpstreamNode', 'us_node': 'UpstreamNode',
        'end': 'DownstreamNode', 'DS': 'DownstreamNode', 'ds_node': 'DownstreamNode',
        'slope': 'Slope', 'Slope': 'Slope',
        'diameter': 'Diameter', 'Diameter': 'Diameter', 'D': 'Diameter',
        'length': 'Length', 'Length': 'Length', 'L': 'Length',
        'manning': 'Manning', 'Manning': 'Manning', 'n': 'Manning',
        'us_x': 'US_X', 'US_X': 'US_X', 'start_x': 'US_X',
        'us_y': 'US_Y', 'US_Y': 'US_Y', 'start_y': 'US_Y',
        'ds_x': 'DS_X', 'DS_X': 'DS_X', 'end_x': 'DS_X',
        'ds_y': 'DS_Y', 'DS_Y': 'DS_Y', 'end_y': 'DS_Y'
    }
    
    df = df.rename(columns=rename_map)
    
    # 验证关键列
    required_cols = ['PipeID', 'UpstreamNode', 'DownstreamNode', 'Slope', 'Diameter', 'Length']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return None, f"缺少关键列: {', '.join(missing)}", False
    
    # 数据类型转换与清洗
    df['PipeID'] = df['PipeID'].astype(str)
    df['UpstreamNode'] = df['UpstreamNode'].astype(str)
    df['DownstreamNode'] = df['DownstreamNode'].astype(str)
    df['Slope'] = pd.to_numeric(df['Slope'], errors='coerce').abs()
    df.loc[df['Slope'] < 0.0001, 'Slope'] = 0.001 # 修正极小坡度
    
    if 'Manning' not in df.columns:
        df['Manning'] = 0.013
    
    # 检查是否有坐标
    has_coords = all(col in df.columns for col in ['US_X', 'US_Y', 'DS_X', 'DS_Y'])
    
    return df, None, has_coords

def convert_coordinates(df):
    """坐标转换 (HK80 -> WGS84)"""
    if not PYPROJ_AVAILABLE:
        return df, "未安装 pyproj 库，无法进行坐标转换。"
    
    # 简单判断是否已经是经纬度 (X < 180)
    if df['US_X'].mean() < 180:
        return df, None 

    try:
        # 假设输入是 HK80 (EPSG:2326)，输出 WGS84 (EPSG:4326)
        transformer = Transformer.from_crs("EPSG:2326", "EPSG:4326", always_xy=True)
        
        us_lon, us_lat = transformer.transform(df['US_X'].values, df['US_Y'].values)
        ds_lon, ds_lat = transformer.transform(df['DS_X'].values, df['DS_Y'].values)
        
        df['US_X_WGS84'] = us_lon
        df['US_Y_WGS84'] = us_lat
        df['DS_X_WGS84'] = ds_lon
        df['DS_Y_WGS84'] = ds_lat
        
        return df, "HK80"
    except Exception as e:
        return df, f"坐标转换失败: {str(e)}"

def build_graph(df):
    """构建网络图并处理环路"""
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['UpstreamNode'], row['DownstreamNode'], 
                   pipe_id=row['PipeID'], length=row['Length'])
    
    cycles_removed = 0
    # 简单的环路移除逻辑：如果不是 DAG，则移除环中的一条边
    if not nx.is_directed_acyclic_graph(G):
        while not nx.is_directed_acyclic_graph(G):
            try:
                cycle = nx.find_cycle(G)
                # 移除环中的第一条边
                G.remove_edge(*cycle[0])
                cycles_removed += 1
            except nx.NetworkXNoCycle:
                break
    return G, cycles_removed

def generate_inflows(nodes, hours=24):
    """生成模拟节点入流数据 (随机模式)"""
    node_inflows = {}
    time_steps = np.arange(hours)
    
    # 固定的随机种子以保证同一文件每次加载结果一致
    np.random.seed(42)
    
    for node in nodes:
        base = np.random.uniform(0.001, 0.005) 
        # 双峰降雨模式
        p1 = np.exp(-((time_steps - 8)**2)/8)
        p2 = np.exp(-((time_steps - 20)**2)/8)
        pattern = 0.5 + 0.5*p1 + 0.4*p2 + np.random.normal(0, 0.05, hours)
        pattern = np.maximum(pattern, 0.1) # 保证非负
        node_inflows[node] = base * pattern
    return node_inflows

# ==========================================
# 3. 模拟逻辑
# ==========================================
def run_simulation_logic(G, df_pipe, hours):
    """执行时变模拟"""
    solver = VectorizedHydraulics()
    
    # 拓扑排序，确定计算顺序
    try:
        topo_nodes = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        st.error("网络中仍存在环路，无法进行水力计算。")
        return None

    all_nodes = list(G.nodes())
    node_inflow_data = generate_inflows(all_nodes, hours=hours)
    
    # 准备矢量化计算所需的数组
    num_pipes = len(df_pipe)
    # 建立 PipeID 到 数组索引 的映射
    pipe_id_to_idx = {pid: i for i, pid in enumerate(df_pipe['PipeID'])}
    
    # 按照 DataFrame 的顺序提取参数
    all_diameters = df_pipe['Diameter'].values
    all_slopes = df_pipe['Slope'].values
    all_mannings = df_pipe['Manning'].values
    
    # 结果容器
    res_Q = np.zeros((num_pipes, hours))
    res_v = np.zeros((num_pipes, hours))
    res_h = np.zeros((num_pipes, hours))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    
    for t in range(hours):
        status_text.text(f"⏳ 正在计算第 {t+1}/{hours} 小时...")
        progress_bar.progress((t + 1) / hours)
        
        # 当前时刻的节点累积流量 (初始为入流)
        node_accumulation = {n: node_inflow_data[n][t] for n in all_nodes}
        
        # 临时存储当前时刻每根管的流量
        current_step_Q = np.zeros(num_pipes)
        
        # 按拓扑顺序传递流量
        for u in topo_nodes:
            total_inflow = node_accumulation[u]
            out_edges = list(G.out_edges(u, data=True))
            
            if not out_edges: continue
            
            # 简单假设：流量平均分配到下游管道
            flow_per_pipe = total_inflow / len(out_edges)
            
            for _, v_node, data in out_edges:
                pid = data['pipe_id']
                if pid in pipe_id_to_idx:
                    idx = pipe_id_to_idx[pid]
                    current_step_Q[idx] = flow_per_pipe
                    
                    # 将流量传递给下游节点
                    if v_node in node_accumulation:
                        node_accumulation[v_node] += flow_per_pipe
        
        # 矢量化求解当前时刻所有管道的水力要素
        h_t, v_t = solver.solve_normal_depth(current_step_Q, all_diameters, all_slopes, all_mannings)
        
        res_Q[:, t] = current_step_Q
        res_v[:, t] = v_t
        res_h[:, t] = h_t
    
    status_text.empty()
    progress_bar.empty()
    
    return {
        'res_Q': res_Q,
        'res_v': res_v,
        'res_h': res_h,
        'duration': time.time() - start_time
    }

# ==========================================
# 4. 界面主逻辑
# ==========================================

st.title("🌊 城市雨水管网水力分析系统")
st.markdown("""
本系统提供基于拓扑网络的稳态/动态水力模拟。
*   **左侧上传**：支持 CSV/Excel 格式的管网数据。
*   **中间地图**：交互式查看管网布局，点击管道选择。
*   **右侧分析**：查看单根管道的水位、流速、流量过程线。
""")

# --- 侧边栏 ---
with st.sidebar:
    st.header("1. 数据导入")
    uploaded_file = st.file_uploader("上传管网数据", type=['xlsx', 'csv'])
    
    with st.expander("数据格式说明"):
        st.markdown("""
        必须包含以下列 (不区分大小写):
        - `PipeID`: 管道编号
        - `UpstreamNode`: 上游节点
        - `DownstreamNode`: 下游节点
        - `Diameter`: 管径 (m)
        - `Length`: 管长 (m)
        - `Slope`: 坡度 (小数)
        - `US_X`, `US_Y`, `DS_X`, `DS_Y`: 起终点坐标 (可选)
        """)

    st.header("2. 模拟参数")
    sim_hours = st.slider("模拟时长 (小时)", 12, 48, 24)
    default_n = st.number_input("默认曼宁系数 (n)", 0.010, 0.020, 0.013, format="%.3f", step=0.001)

# --- 主体逻辑 ---
if uploaded_file:
    df_pipe, error_msg, has_coords = load_and_process_data(uploaded_file)
    
    if error_msg:
        st.error(error_msg)
    else:
        # 应用默认曼宁系数
        if 'Manning' not in df_pipe.columns:
            df_pipe['Manning'] = default_n
            
        # 构建图
        G, cycles_removed = build_graph(df_pipe)
        if cycles_removed > 0:
            st.warning(f"⚠️ 检测到管网中存在环路，已自动断开 {cycles_removed} 处连接以构建有向无环图(DAG)。")

        # 布局
        col_map, col_details = st.columns([1.6, 1])
        
        # ... (前文代码保持不变) ...

        # --- 地图区域 (修改了点击处理逻辑) ---
        with col_map:
            st.subheader("🗺️ 管网地图")
            
            # ... (地图数据准备代码保持不变) ...
            
            # 渲染地图
            selection = st.pydeck_chart(
                deck, 
                on_select="rerun", 
                selection_mode="single-object",
                use_container_width=True
            )
            
            # ★★★ 修复点 1: 地图点击逻辑 ★★★
            # 只有当确实发生了点击，且点击的 ID 与当前不同时，才更新状态
            if selection.selection:
                indices = selection.selection.get("indices")
                if indices:
                    clicked_idx = indices[0]
                    clicked_id = df_map.iloc[clicked_idx]['PipeID']
                    
                    # 只有当点击的新管道与当前存储的不一致时，才更新
                    # 注意：这里我们更新 'pipe_selector'，这是绑定给 selectbox 的 key
                    if clicked_id != st.session_state.get('pipe_selector'):
                        st.session_state['pipe_selector'] = clicked_id
                        st.session_state['selected_pipe_id'] = clicked_id
                        st.rerun()

        # --- 详情与图表区域 (修改了下拉框逻辑) ---
        with col_details:
            st.subheader("📈 模拟与分析")
            
            # ... (模拟控制按钮代码保持不变) ...

            st.divider()

            # 管道选择器逻辑
            all_ids = df_pipe['PipeID'].values.tolist()
            
            # 初始化默认值
            if 'pipe_selector' not in st.session_state or st.session_state['pipe_selector'] not in all_ids:
                if all_ids:
                    st.session_state['pipe_selector'] = all_ids[0]
                    st.session_state['selected_pipe_id'] = all_ids[0]

            # ★★★ 修复点 2: 下拉框回调函数 ★★★
            def on_selector_change():
                # 当用户手动改变下拉框时，同步更新 selected_pipe_id
                st.session_state['selected_pipe_id'] = st.session_state['pipe_selector']

            # ★★★ 修复点 3: 下拉框配置 ★★★
            # 1. key='pipe_selector': 直接绑定 Session State
            # 2. index: 显式计算当前 ID 在列表中的位置，确保 UI 显示正确
            try:
                current_index = all_ids.index(st.session_state['pipe_selector'])
            except ValueError:
                current_index = 0

            selected_id = st.selectbox(
                "选择管段查看详情:", 
                options=all_ids,
                key="pipe_selector",  # 关键：双向绑定
                index=current_index,  # 关键：确保显示正确
                on_change=on_selector_change
            )
            
            # 确保 selected_pipe_id 与下拉框一致 (双重保险)
            st.session_state['selected_pipe_id'] = selected_id

            # ... (后续显示属性和图表的代码保持不变) ...
            
            # 模拟控制
            sim_params_changed = (
                st.session_state['simulation_params'].get('hours') != sim_hours or
                st.session_state['simulation_params'].get('n') != default_n
            )
            
            if not st.session_state['has_results'] or sim_params_changed:
                st.info("参数已更新或尚未计算，请运行模拟。")
                if st.button("🚀 开始模拟计算", type="primary", use_container_width=True):
                    results = run_simulation_logic(G, df_pipe, sim_hours)
                    if results:
                        st.session_state['res_Q'] = results['res_Q']
                        st.session_state['res_v'] = results['res_v']
                        st.session_state['res_h'] = results['res_h']
                        st.session_state['all_pipe_ids'] = df_pipe['PipeID'].values
                        st.session_state['has_results'] = True
                        st.session_state['simulation_params'] = {'hours': sim_hours, 'n': default_n}
                        st.success(f"计算完成！耗时 {results['duration']:.2f}s")
                        st.rerun()
            else:
                st.success("✅ 模拟结果已就绪")
                if st.button("🔄 重新计算", use_container_width=True):
                    # 清除结果触发重算
                    st.session_state['has_results'] = False
                    st.rerun()

            st.divider()

            # 管道选择器
            all_ids = df_pipe['PipeID'].values.tolist()
            
            # 确保默认选中第一个
            if st.session_state['selected_pipe_id'] is None and all_ids:
                st.session_state['selected_pipe_id'] = all_ids[0]
                st.session_state['pipe_selector'] = all_ids[0]

            # 回调函数：下拉框变动时更新 selected_pipe_id
            def on_selector_change():
                st.session_state['selected_pipe_id'] = st.session_state['pipe_selector']

            # 下拉框
            selected_id = st.selectbox(
                "选择管段查看详情:", 
                options=all_ids,
                key="pipe_selector",
                index=all_ids.index(st.session_state['selected_pipe_id']) if st.session_state['selected_pipe_id'] in all_ids else 0,
                on_change=on_selector_change
            )
            
            # 显示属性
            pipe_row = df_pipe[df_pipe['PipeID'] == selected_id]
            if not pipe_row.empty:
                info = pipe_row.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("管径", f"{info['Diameter']} m")
                c2.metric("管长", f"{info['Length']} m")
                c3.metric("坡度", f"{info['Slope']:.4f}")
                
                # 显示图表
                if st.session_state['has_results']:
                    try:
                        # 找到该管段在结果数组中的索引
                        idx = np.where(st.session_state['all_pipe_ids'] == selected_id)[0][0]
                        
                        ts_Q = st.session_state['res_Q'][idx, :]
                        ts_v = st.session_state['res_v'][idx, :]
                        ts_h = st.session_state['res_h'][idx, :]
                        hours_arr = np.arange(sim_hours)
                        
                        if PLOTLY_AVAILABLE:
                            fig = make_subplots(
                                rows=3, cols=1, 
                                shared_xaxes=True, 
                                vertical_spacing=0.05,
                                subplot_titles=("流量 Q (m³/s)", "流速 v (m/s)", "水深 h (m)")
                            )
                            
                            # 样式配置
                            line_style = dict(width=2)
                            
                            fig.add_trace(go.Scatter(
                                x=hours_arr, y=ts_Q, mode='lines', name="流量",
                                line=dict(color='#3b82f6', **line_style), 
                                fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'
                            ), row=1, col=1)
                            
                            fig.add_trace(go.Scatter(
                                x=hours_arr, y=ts_v, mode='lines', name="流速",
                                line=dict(color='#f97316', **line_style)
                            ), row=2, col=1)
                            
                            fig.add_trace(go.Scatter(
                                x=hours_arr, y=ts_h, mode='lines', name="水深",
                                line=dict(color='#22c55e', **line_style), 
                                fill='tozeroy', fillcolor='rgba(34, 197, 94, 0.2)'
                            ), row=3, col=1)
                            
                            # 警戒线
                            fig.add_hline(y=info['Diameter'], line_dash="dash", line_color="red", 
                                          annotation_text="管顶", row=3, col=1)

                            fig.update_layout(
                                height=500, 
                                margin=dict(l=0, r=0, t=20, b=0), 
                                showlegend=False,
                                hovermode="x unified",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.line_chart(pd.DataFrame({'Q': ts_Q, 'v': ts_v, 'h': ts_h}))
                            
                    except Exception as e:
                        st.error(f"无法获取结果数据: {e}")
                else:
                    st.info("等待模拟结果...")

else:
    st.info("👈 请先在左侧侧边栏上传数据文件。")

