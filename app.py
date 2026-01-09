# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pydeck as pdk
import warnings
import time

# 尝试导入 pyproj 用于坐标转换
try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

# 尝试导入 plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# 设置页面配置
st.set_page_config(page_title="城市雨水管网水力模拟系统", layout="wide")

# ==========================================
# 0. 初始化 Session State
# ==========================================
# 我们需要两个状态：
# 1. selected_pipe_id: 存储当前选中的真实 PipeID
# 2. pipe_selector: 绑定到下拉框的 key，用于强制更新 UI
if 'selected_pipe_id' not in st.session_state:
    st.session_state['selected_pipe_id'] = None

# 忽略警告
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 核心水力计算类 (保持不变)
# ==========================================
class VectorizedHydraulics:
    def solve_normal_depth(self, Q_target, D, S, n):
        S = np.where(S <= 1e-6, 1e-6, S)
        sqrt_S = np.sqrt(S)
        Q_full_capacity = (1/n) * (np.pi*(D/2)**2) * ((D/4)**(2/3)) * sqrt_S
        K_target = (Q_target * n) / sqrt_S
        overloaded = Q_target >= Q_full_capacity
        
        theta = np.full_like(Q_target, np.pi, dtype=np.float64)
        mask_solve = (~overloaded) & (Q_target > 0.0001)
        
        if not np.any(mask_solve):
             h = np.zeros_like(Q_target)
             h[overloaded] = D[overloaded]
             v = np.zeros_like(Q_target)
             full_area = np.pi * (D/2)**2
             v[overloaded] = Q_target[overloaded] / full_area[overloaded]
             return h, v

        theta_active = theta[mask_solve]
        D_active = D[mask_solve]
        K_t_active = K_target[mask_solve]
        coef_active = (D_active**2) / 8
        
        for _ in range(8):
            sin_t = np.sin(theta_active)
            cos_t = np.cos(theta_active)
            A = coef_active * (theta_active - sin_t)
            P = (D_active / 2) * theta_active
            P[P < 1e-6] = 1e-6
            R = A / P
            f_val = A * (R**(2/3)) - K_t_active
            dA_dth = coef_active * (1 - cos_t)
            dP_dth = D_active / 2
            term1 = (5/3) * (A**(2/3)) * (P**(-2/3)) * dA_dth
            term2 = (2/3) * (A**(5/3)) * (P**(-5/3)) * dP_dth
            f_prime = term1 - term2
            f_prime[np.abs(f_prime) < 1e-6] = 1e-6
            theta_active -= f_val / f_prime
            theta_active = np.clip(theta_active, 1e-4, 2*np.pi - 1e-4)

        theta[mask_solve] = theta_active
        theta[overloaded] = 2 * np.pi
        theta[Q_target <= 0.0001] = 0
        
        h = (D / 2) * (1 - np.cos(theta / 2))
        A_final = (D**2 / 8) * (theta - np.sin(theta))
        v = np.zeros_like(Q_target)
        valid_A = A_final > 1e-6
        v[valid_A] = Q_target[valid_A] / A_final[valid_A]
        
        return h, v

# ==========================================
# 2. 数据处理
# ==========================================
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    rename_map = {
        'name': 'PipeID', 'Pipe': 'PipeID', 'pipe_id': 'PipeID',
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
    required_cols = ['PipeID', 'UpstreamNode', 'DownstreamNode', 'Slope', 'Diameter', 'Length']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return None, f"缺少关键列: {missing}", False
    
    has_coords = all(col in df.columns for col in ['US_X', 'US_Y', 'DS_X', 'DS_Y'])
    df['UpstreamNode'] = df['UpstreamNode'].astype(str)
    df['DownstreamNode'] = df['DownstreamNode'].astype(str)
    df['Slope'] = pd.to_numeric(df['Slope'], errors='coerce').abs()
    df.loc[df['Slope'] < 0.0001, 'Slope'] = 0.001
    
    if 'Manning' not in df.columns:
        df['Manning'] = 0.013
    
    # 确保 PipeID 是字符串
    df['PipeID'] = df['PipeID'].astype(str)
        
    return df, None, has_coords

def convert_coordinates(df):
    if not PYPROJ_AVAILABLE:
        return df, "未安装 pyproj 库，无法进行坐标转换。"
    
    # 简单的判断：如果坐标很小，可能已经是经纬度
    if df['US_X'].mean() < 180:
        return df, None 

    try:
        transformer = Transformer.from_crs("EPSG:2326", "EPSG:4326", always_xy=True)
        us_lon, us_lat = transformer.transform(df['US_X'].values, df['US_Y'].values)
        df['US_X_WGS84'] = us_lon
        df['US_Y_WGS84'] = us_lat
        ds_lon, ds_lat = transformer.transform(df['DS_X'].values, df['DS_Y'].values)
        df['DS_X_WGS84'] = ds_lon
        df['DS_Y_WGS84'] = ds_lat
        return df, "HK80"
    except Exception as e:
        return df, f"坐标转换失败: {str(e)}"

def build_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['UpstreamNode'], row['DownstreamNode'], pipe_id=row['PipeID'], length=row['Length'])
    cycles_removed = 0
    if not nx.is_directed_acyclic_graph(G):
        while not nx.is_directed_acyclic_graph(G):
            try:
                cycle = nx.find_cycle(G)
                G.remove_edge(*cycle[0])
                cycles_removed += 1
            except:
                break
    return G, cycles_removed

def generate_inflows(nodes, hours=24):
    node_inflows = {}
    time_steps = np.arange(hours)
    for node in nodes:
        base = np.random.uniform(0.001, 0.005) 
        p1 = np.exp(-((time_steps - 8)**2)/8)
        p2 = np.exp(-((time_steps - 20)**2)/8)
        pattern = 0.5 + 0.5*p1 + 0.4*p2 + np.random.normal(0, 0.05, hours)
        pattern = np.maximum(pattern, 0.1)
        node_inflows[node] = base * pattern
    return node_inflows

# ==========================================
# 3. Streamlit 界面逻辑
# ==========================================

st.title("🌊 城市雨水管网水力分析系统 (Web版)")
st.markdown("支持香港1980坐标系 (HK80) 自动转换为地图经纬度。**点击地图上的管段可查看详情。**")

# --- 侧边栏 ---
st.sidebar.header("1. 数据导入")
uploaded_file = st.sidebar.file_uploader("上传文件", type=['xlsx', 'csv'])

st.sidebar.header("2. 模拟参数")
sim_hours = st.sidebar.slider("模拟时长 (小时)", 12, 48, 24)
default_n = st.sidebar.number_input("默认曼宁系数", 0.010, 0.020, 0.013, format="%.3f")

# 运行模拟的函数
def run_simulation(G, df_pipe, hours):
    solver = VectorizedHydraulics()
    topo_nodes = list(nx.topological_sort(G))
    all_nodes = list(G.nodes())
    node_inflow_data = generate_inflows(all_nodes, hours=hours)
    
    num_pipes = len(df_pipe)
    all_pipe_ids = df_pipe['PipeID'].values
    all_diameters = df_pipe['Diameter'].values
    all_slopes = df_pipe['Slope'].values
    all_mannings = df_pipe['Manning'].values
    
    res_Q = np.zeros((num_pipes, hours))
    res_v = np.zeros((num_pipes, hours))
    res_h = np.zeros((num_pipes, hours))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    for t in range(hours):
        status_text.text(f"计算进度: {t+1}/{hours} 小时")
        progress_bar.progress((t + 1) / hours)
        
        node_accumulation = {n: node_inflow_data[n][t] for n in all_nodes}
        pipe_flow_snapshot = {}
        
        for u in topo_nodes:
            total_inflow = node_accumulation[u]
            out_edges = list(G.out_edges(u, data=True))
            if not out_edges: continue
            
            flow_per_pipe = total_inflow / len(out_edges)
            for _, v_node, data in out_edges:
                pid = data['pipe_id']
                pipe_flow_snapshot[pid] = flow_per_pipe
                if v_node in node_accumulation:
                    node_accumulation[v_node] += flow_per_pipe
        
        current_Q = np.array([pipe_flow_snapshot.get(pid, 0.0) for pid in all_pipe_ids])
        h_t, v_t = solver.solve_normal_depth(current_Q, all_diameters, all_slopes, all_mannings)
        
        res_Q[:, t] = current_Q
        res_v[:, t] = v_t
        res_h[:, t] = h_t
    
    status_text.empty()
    progress_bar.empty()
    
    st.session_state['res_Q'] = res_Q
    st.session_state['res_v'] = res_v
    st.session_state['res_h'] = res_h
    st.session_state['all_pipe_ids'] = all_pipe_ids
    st.session_state['all_diameters'] = all_diameters
    st.session_state['has_results'] = True
    st.success(f"计算完成！耗时 {time.time() - start_time:.2f} 秒")

if uploaded_file:
    df_pipe, error_msg, has_coords = load_data(uploaded_file)
    
    if error_msg:
        st.error(error_msg)
    else:
        if 'Manning' not in df_pipe.columns:
            df_pipe['Manning'] = default_n
            
        G, cycles = build_graph(df_pipe)
        
        # 布局：左侧地图，右侧/下方详情
        col_map, col_details = st.columns([1.5, 1])
        
        # 准备地图数据（放在列外面，因为后续逻辑需要用到 df_map）
        df_map = df_pipe.copy()
        if has_coords:
            df_map, trans_status = convert_coordinates(df_map)
            # ★★★ 关键修复：重置索引，确保 pydeck 返回的 index 能对应上 ★★★
            df_map = df_map.reset_index(drop=True)
            
            if trans_status == "HK80":
                x_col_us, y_col_us = 'US_X_WGS84', 'US_Y_WGS84'
                x_col_ds, y_col_ds = 'DS_X_WGS84', 'DS_Y_WGS84'
            else:
                x_col_us, y_col_us = 'US_X', 'US_Y'
                x_col_ds, y_col_ds = 'DS_X', 'DS_Y'

            d_min, d_max = df_map['Diameter'].min(), df_map['Diameter'].max()
            def get_color(d):
                if d_max == d_min: ratio = 0.5
                else: ratio = (d - d_min) / (d_max - d_min)
                r = int(255 * ratio)
                g = int(255 * (1 - ratio))
                return [r, g, 0, 200]
            df_map['color'] = df_map['Diameter'].apply(get_color)

        with col_map:
            st.subheader("🗺️ GIS 管网交互地图")
            if has_coords:
                mid_lat = (df_map[y_col_us].mean() + df_map[y_col_ds].mean()) / 2
                mid_lon = (df_map[x_col_us].mean() + df_map[x_col_ds].mean()) / 2

                layer = pdk.Layer(
                    "LineLayer",
                    df_map, # 这里传入的是已经 reset_index 的数据
                    get_source_position=[x_col_us, y_col_us],
                    get_target_position=[x_col_ds, y_col_ds],
                    get_color="color",
                    get_width=5,
                    pickable=True,
                    auto_highlight=True,
                )

                view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=13, pitch=0)

                deck = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    map_style='mapbox://styles/mapbox/dark-v10',
                    tooltip={"text": "点击查看详情\nID: {PipeID}"}
                )
                
                # 渲染地图
                selection = st.pydeck_chart(
                    deck, 
                    on_select="rerun", 
                    selection_mode="single-object",
                    use_container_width=True
                )
                
                # ★★★ 关键修复：处理点击逻辑 ★★★
                if selection.selection:
                    indices = selection.selection.get("indices")
                    if indices:
                        clicked_index = indices[0]
                        # 使用 reset_index 后的 df_map 获取 ID
                        clicked_pipe_id = df_map.iloc[clicked_index]['PipeID']
                        
                        # 如果点击了新的管段，更新 session state 并强制刷新
                        if clicked_pipe_id != st.session_state['selected_pipe_id']:
                            st.session_state['selected_pipe_id'] = clicked_pipe_id
                            # 同时更新下拉框的 key，确保 UI 同步
                            st.session_state['pipe_selector'] = clicked_pipe_id
                            st.rerun()
            else:
                st.warning("无坐标数据，无法显示地图")

        with col_details:
            st.subheader("📊 模拟与分析")
            
            # 1. 模拟控制
            if not st.session_state.get('has_results', False):
                st.info("尚未运行模拟。点击下方按钮开始计算。")
                if st.button("🚀 开始模拟计算", type="primary"):
                    run_simulation(G, df_pipe, sim_hours)
                    st.rerun()
            else:
                if st.button("🔄 重新运行模拟"):
                    run_simulation(G, df_pipe, sim_hours)
                    st.rerun()

            st.divider()

            # 2. 结果展示
            # 如果还没有选中任何管段，默认选第一个
            if st.session_state['selected_pipe_id'] is None and len(df_pipe) > 0:
                first_id = df_pipe.iloc[0]['PipeID']
                st.session_state['selected_pipe_id'] = first_id
                # 初始化下拉框的 key
                if 'pipe_selector' not in st.session_state:
                    st.session_state['pipe_selector'] = first_id

            all_ids = df_pipe['PipeID'].values.tolist()
            
            # 回调函数：当用户手动改变下拉框时触发
            def on_pipe_select_change():
                st.session_state['selected_pipe_id'] = st.session_state['pipe_selector']

            # 下拉框
            # key='pipe_selector' 绑定了 session_state 中的值
            # 当地图点击更新了 session_state['pipe_selector'] 后，这个下拉框会自动跳到对应的值
            selected_pipe = st.selectbox(
                "当前选中管段:", 
                all_ids, 
                key="pipe_selector",
                on_change=on_pipe_select_change
            )
            
            # 确保 selected_pipe_id 与下拉框保持一致 (双重保险)
            current_pipe_id = selected_pipe

            # 展示选中管段的静态属性
            pipe_row = df_pipe[df_pipe['PipeID'] == current_pipe_id]
            if not pipe_row.empty:
                pipe_info = pipe_row.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("管径", f"{pipe_info['Diameter']} m")
                c2.metric("长度", f"{pipe_info['Length']} m")
                c3.metric("坡度", f"{pipe_info['Slope']:.4f}")

                # 展示动态结果
                if st.session_state.get('has_results', False):
                    try:
                        idx = np.where(st.session_state['all_pipe_ids'] == current_pipe_id)[0][0]
                        ts_Q = st.session_state['res_Q'][idx, :]
                        ts_v = st.session_state['res_v'][idx, :]
                        ts_h = st.session_state['res_h'][idx, :]
                        hours_arr = np.arange(sim_hours)
                        
                        if PLOTLY_AVAILABLE:
                            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                                vertical_spacing=0.05,
                                                subplot_titles=("流量 Q (m³/s)", "流速 v (m/s)", "水深 h (m)"))
                            fig.add_trace(go.Scatter(x=hours_arr, y=ts_Q, name="流量", line=dict(color='#3b82f6')), row=1, col=1)
                            fig.add_trace(go.Scatter(x=hours_arr, y=ts_v, name="流速", line=dict(color='#f97316')), row=2, col=1)
                            fig.add_trace(go.Scatter(x=hours_arr, y=ts_h, name="水深", line=dict(color='#22c55e'), fill='tozeroy'), row=3, col=1)
                            fig.add_hline(y=pipe_info['Diameter'], line_dash="dash", line_color="red", annotation_text="管顶", row=3, col=1)
                            fig.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.line_chart(pd.DataFrame({'Q': ts_Q, 'v': ts_v, 'h': ts_h}))
                    except IndexError:
                        st.error("未找到该管段的模拟结果，请重新运行模拟。")
                else:
                    st.info("👆 请先点击上方的“开始模拟计算”按钮查看水力结果。")
            else:
                st.error("数据中找不到该管段ID")

else:
    st.info("请在左侧上传数据文件。")
    st.markdown("""
    **文件列名说明（不区分大小写）：**
    - 坐标：`us_x`, `us_y` (上游); `ds_x`, `ds_y` (下游)
    - 属性：`PipeID`, `Diameter`, `Slope`, `Length`
    """)
