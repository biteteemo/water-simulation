# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
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
if 'selected_pipe_id' not in st.session_state:
    st.session_state['selected_pipe_id'] = None
if 'pipe_selector' not in st.session_state:
    st.session_state['pipe_selector'] = None

# 忽略警告
warnings.filterwarnings('ignore')

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
    
    df['PipeID'] = df['PipeID'].astype(str)
        
    return df, None, has_coords

def convert_coordinates(df):
    if not PYPROJ_AVAILABLE:
        return df, "未安装 pyproj 库，无法进行坐标转换。"
    
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

st.title("🌊 城市雨水管网水力分析系统")
st.markdown("👉 **鼠标悬停**在管道上可高亮显示；**点击**管道可在右侧查看详细水力曲线。")

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
        
        # 布局：左侧地图，右侧详情
        col_map, col_details = st.columns([1.6, 1])
        
        # 准备地图数据
        df_map = df_pipe.copy()
        if has_coords:
            df_map, trans_status = convert_coordinates(df_map)
            # ★★★ 关键：重置索引，确保 pydeck 返回的 index 能对应上 ★★★
            df_map = df_map.reset_index(drop=True)
            
            if trans_status == "HK80":
                x_col_us, y_col_us = 'US_X_WGS84', 'US_Y_WGS84'
                x_col_ds, y_col_ds = 'DS_X_WGS84', 'DS_Y_WGS84'
            else:
                x_col_us, y_col_us = 'US_X', 'US_Y'
                x_col_ds, y_col_ds = 'DS_X', 'DS_Y'

            # 颜色映射
            d_min, d_max = df_map['Diameter'].min(), df_map['Diameter'].max()
            def get_color(d):
                if d_max == d_min: ratio = 0.5
                else: ratio = (d - d_min) / (d_max - d_min)
                r = int(50 + 205 * ratio)
                g = int(50 + 205 * (1 - ratio))
                return [r, g, 200] # 基础颜色
            
            df_map['color'] = df_map['Diameter'].apply(get_color)

        with col_map:
            st.subheader("🗺️ 交互地图 (点击选择)")
            if has_coords:
                mid_lat = (df_map[y_col_us].mean() + df_map[y_col_ds].mean()) / 2
                mid_lon = (df_map[x_col_us].mean() + df_map[x_col_ds].mean()) / 2

                # ★★★ 核心修改：增强交互体验 ★★★
                layer = pdk.Layer(
                    "LineLayer",
                    df_map,
                    get_source_position=[x_col_us, y_col_us],
                    get_target_position=[x_col_ds, y_col_ds],
                    get_color="color",
                    get_width=8,  # 加粗线条，方便鼠标悬停
                    pickable=True, # 允许交互（悬停和点击）
                    auto_highlight=True, # 开启自动高亮
                    highlight_color=[255, 255, 0, 255], # 悬停时显示亮黄色
                )

                view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=13, pitch=0)

                deck = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    map_style='mapbox://styles/mapbox/dark-v10',
                    # 悬停时显示的提示框
                    tooltip={
                        "html": "<b>Pipe ID:</b> {PipeID}<br/><b>管径:</b> {Diameter}m<br/><b>点击查看详情</b>",
                        "style": {"backgroundColor": "steelblue", "color": "white"}
                    }
                )
                
                # 渲染地图，监听点击事件
                selection = st.pydeck_chart(
                    deck, 
                    on_select="rerun", 
                    selection_mode="single-object",
                    use_container_width=True
                )
                
                # 处理点击逻辑
                if selection.selection:
                    indices = selection.selection.get("indices")
                    if indices:
                        clicked_index = indices[0]
                        clicked_pipe_id = df_map.iloc[clicked_index]['PipeID']
                        
                        if clicked_pipe_id != st.session_state['selected_pipe_id']:
                            st.session_state['selected_pipe_id'] = clicked_pipe_id
                            st.session_state['pipe_selector'] = clicked_pipe_id
                            st.rerun()
            else:
                st.warning("无坐标数据，无法显示地图")

        with col_details:
            st.subheader("📈 管段详情与曲线")
            
            # 模拟控制按钮
            if not st.session_state.get('has_results', False):
                st.info("请先点击下方按钮进行水力计算")
                if st.button("🚀 开始模拟计算", type="primary"):
                    run_simulation(G, df_pipe, sim_hours)
                    st.rerun()
            else:
                if st.button("🔄 重新计算"):
                    run_simulation(G, df_pipe, sim_hours)
                    st.rerun()

            st.divider()

            # 默认选中逻辑
            if st.session_state['selected_pipe_id'] is None and len(df_pipe) > 0:
                first_id = df_pipe.iloc[0]['PipeID']
                st.session_state['selected_pipe_id'] = first_id
                if 'pipe_selector' not in st.session_state or st.session_state['pipe_selector'] is None:
                    st.session_state['pipe_selector'] = first_id

            # 下拉框同步
            all_ids = df_pipe['PipeID'].values.tolist()
            def on_pipe_select_change():
                st.session_state['selected_pipe_id'] = st.session_state['pipe_selector']

            selected_pipe = st.selectbox(
                "当前选中管段 (也可在地图点击):", 
                all_ids, 
                key="pipe_selector",
                on_change=on_pipe_select_change
            )
            
            current_pipe_id = selected_pipe

            # 显示静态属性
            pipe_row = df_pipe[df_pipe['PipeID'] == current_pipe_id]
            if not pipe_row.empty:
                pipe_info = pipe_row.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("管径 (D)", f"{pipe_info['Diameter']} m")
                c2.metric("管长 (L)", f"{pipe_info['Length']} m")
                c3.metric("坡度 (S)", f"{pipe_info['Slope']:.4f}")

                # 显示动态曲线
                if st.session_state.get('has_results', False):
                    try:
                        idx = np.where(st.session_state['all_pipe_ids'] == current_pipe_id)[0][0]
                        ts_Q = st.session_state['res_Q'][idx, :]
                        ts_v = st.session_state['res_v'][idx, :]
                        ts_h = st.session_state['res_h'][idx, :]
                        hours_arr = np.arange(sim_hours)
                        
                        if PLOTLY_AVAILABLE:
                            # 创建三个子图
                            fig = make_subplots(
                                rows=3, cols=1, 
                                shared_xaxes=True, 
                                vertical_spacing=0.08,
                                subplot_titles=("流量 Q (m³/s)", "流速 v (m/s)", "水深 h (m)")
                            )
                            
                            # 1. 流量
                            fig.add_trace(go.Scatter(
                                x=hours_arr, y=ts_Q, mode='lines', name="流量",
                                line=dict(color='#3b82f6', width=2), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'
                            ), row=1, col=1)
                            
                            # 2. 流速
                            fig.add_trace(go.Scatter(
                                x=hours_arr, y=ts_v, mode='lines', name="流速",
                                line=dict(color='#f97316', width=2)
                            ), row=2, col=1)
                            
                            # 3. 水深
                            fig.add_trace(go.Scatter(
                                x=hours_arr, y=ts_h, mode='lines', name="水深",
                                line=dict(color='#22c55e', width=2), fill='tozeroy', fillcolor='rgba(34, 197, 94, 0.2)'
                            ), row=3, col=1)
                            
                            # 添加管顶警戒线
                            fig.add_hline(y=pipe_info['Diameter'], line_dash="dash", line_color="red", 
                                          annotation_text="管顶", row=3, col=1)

                            fig.update_layout(
                                height=600, 
                                margin=dict(l=10, r=10, t=30, b=10), 
                                showlegend=False,
                                hovermode="x unified"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("未检测到 Plotly 库，使用基础图表展示")
                            st.line_chart(pd.DataFrame({'Q': ts_Q, 'v': ts_v, 'h': ts_h}))
                    except IndexError:
                        st.error("结果索引错误，请重新运行模拟。")
                else:
                    st.info("👆 数据已就绪，请点击“开始模拟计算”查看曲线。")
            else:
                st.error("数据中找不到该管段ID")

else:
    st.info("请在左侧上传数据文件。")
