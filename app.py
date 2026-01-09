# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import pydeck as pdk
import warnings
import time
import math

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
    'has_results': False,
    'res_Q': None,
    'res_v': None,
    'res_h': None,
    'all_pipe_ids': None,
    'simulation_params': {},
}

for key, val in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==========================================
# 1. 核心计算类 (保持不变)
# ==========================================
class VectorizedHydraulics:
    def solve_normal_depth(self, Q_target, D, S, n):
        S = np.where(S <= 1e-6, 1e-6, S)
        sqrt_S = np.sqrt(S)
        Q_full_capacity = (1/n) * (np.pi*(D/2)**2) * ((D/4)**(2/3)) * sqrt_S
        overloaded = Q_target >= Q_full_capacity
        K_target = (Q_target * n) / sqrt_S
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
# 2. 数据处理函数
# ==========================================
@st.cache_data
def load_and_process_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        return None, f"文件读取失败: {str(e)}", False

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
    required_cols = ['PipeID', 'UpstreamNode', 'DownstreamNode', 'Slope', 'Diameter', 'Length']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return None, f"缺少关键列: {', '.join(missing)}", False
    
    # 强制转换为字符串，避免类型混淆
    df['PipeID'] = df['PipeID'].astype(str)
    df['UpstreamNode'] = df['UpstreamNode'].astype(str)
    df['DownstreamNode'] = df['DownstreamNode'].astype(str)
    
    df['Slope'] = pd.to_numeric(df['Slope'], errors='coerce').abs()
    df.loc[df['Slope'] < 0.0001, 'Slope'] = 0.001
    
    if 'Manning' not in df.columns:
        df['Manning'] = 0.013
    
    has_coords = all(col in df.columns for col in ['US_X', 'US_Y', 'DS_X', 'DS_Y'])
    return df, None, has_coords

def convert_coordinates(df):
    if not PYPROJ_AVAILABLE:
        return df, "未安装 pyproj 库"
    if df['US_X'].mean() < 180:
        return df, None 
    try:
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
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['UpstreamNode'], row['DownstreamNode'], 
                   pipe_id=row['PipeID'], length=row['Length'])
    cycles_removed = 0
    if not nx.is_directed_acyclic_graph(G):
        while not nx.is_directed_acyclic_graph(G):
            try:
                cycle = nx.find_cycle(G)
                G.remove_edge(*cycle[0])
                cycles_removed += 1
            except nx.NetworkXNoCycle:
                break
    return G, cycles_removed

def generate_inflows(nodes, hours=24):
    node_inflows = {}
    time_steps = np.arange(hours)
    np.random.seed(42)
    for node in nodes:
        base = np.random.uniform(0.001, 0.005) 
        p1 = np.exp(-((time_steps - 8)**2)/8)
        p2 = np.exp(-((time_steps - 20)**2)/8)
        pattern = 0.5 + 0.5*p1 + 0.4*p2 + np.random.normal(0, 0.05, hours)
        pattern = np.maximum(pattern, 0.1)
        node_inflows[node] = base * pattern
    return node_inflows

# ==========================================
# 3. 模拟逻辑
# ==========================================
def run_simulation_logic(G, df_pipe, hours):
    solver = VectorizedHydraulics()
    try:
        topo_nodes = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        st.error("网络中仍存在环路，无法进行水力计算。")
        return None

    all_nodes = list(G.nodes())
    node_inflow_data = generate_inflows(all_nodes, hours=hours)
    
    num_pipes = len(df_pipe)
    pipe_id_to_idx = {pid: i for i, pid in enumerate(df_pipe['PipeID'])}
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
        status_text.text(f"⏳ 正在计算第 {t+1}/{hours} 小时...")
        progress_bar.progress((t + 1) / hours)
        node_accumulation = {n: node_inflow_data[n][t] for n in all_nodes}
        current_step_Q = np.zeros(num_pipes)
        
        for u in topo_nodes:
            total_inflow = node_accumulation[u]
            out_edges = list(G.out_edges(u, data=True))
            if not out_edges: continue
            flow_per_pipe = total_inflow / len(out_edges)
            for _, v_node, data in out_edges:
                pid = data['pipe_id']
                if pid in pipe_id_to_idx:
                    idx = pipe_id_to_idx[pid]
                    current_step_Q[idx] = flow_per_pipe
                    if v_node in node_accumulation:
                        node_accumulation[v_node] += flow_per_pipe
        
        h_t, v_t = solver.solve_normal_depth(current_step_Q, all_diameters, all_slopes, all_mannings)
        res_Q[:, t] = current_step_Q
        res_v[:, t] = v_t
        res_h[:, t] = h_t
    
    status_text.empty()
    progress_bar.empty()
    
    return {
        'res_Q': res_Q, 'res_v': res_v, 'res_h': res_h,
        'duration': time.time() - start_time
    }

# ==========================================
# 4. 详情显示组件
# ==========================================
def render_pipe_details(pipe_id, df_pipe, sim_hours):
    """
    在当前容器中渲染管道详情
    """
    # 确保类型一致
    pipe_id = str(pipe_id)
    pipe_row = df_pipe[df_pipe['PipeID'] == pipe_id]
    
    if pipe_row.empty:
        st.error(f"未找到管道 ID: {pipe_id}")
        return

    info = pipe_row.iloc[0]
    
    st.markdown(f"### 📍 管道: {pipe_id}")
    st.divider()
    
    st.markdown("#### 📌 基础属性")
    c1, c2 = st.columns(2)
    c1.metric("管径", f"{info['Diameter']} m")
    c2.metric("管长", f"{info['Length']} m")
    c1.metric("坡度", f"{info['Slope']:.4f}")
    c2.metric("曼宁", f"{info['Manning']}")
    
    st.divider()
    
    if st.session_state['has_results']:
        try:
            # 查找结果索引 (确保类型匹配)
            all_ids = st.session_state['all_pipe_ids'].astype(str)
            idx = np.where(all_ids == pipe_id)[0]
            
            if len(idx) == 0:
                st.warning("该管道不在计算结果中")
                return
            idx = idx[0]

            ts_Q = st.session_state['res_Q'][idx, :]
            ts_v = st.session_state['res_v'][idx, :]
            ts_h = st.session_state['res_h'][idx, :]
            hours_arr = np.arange(sim_hours)
            
            avg_Q = np.mean(ts_Q)
            avg_v = np.mean(ts_v)
            avg_h = np.mean(ts_h)
            
            st.markdown("#### 📊 模拟统计 (平均值)")
            m1, m2 = st.columns(2)
            m1.metric("流量", f"{avg_Q:.4f} m³/s")
            m2.metric("流速", f"{avg_v:.4f} m/s")
            m1.metric("水深", f"{avg_h:.4f} m")
            
            st.markdown("#### 📉 过程线")
            if PLOTLY_AVAILABLE:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                    subplot_titles=("流量 Q", "流速 v", "水深 h"))
                line_style = dict(width=2)
                fig.add_trace(go.Scatter(x=hours_arr, y=ts_Q, name="流量", line=dict(color='#3b82f6', **line_style), fill='tozeroy'), row=1, col=1)
                fig.add_trace(go.Scatter(x=hours_arr, y=ts_v, name="流速", line=dict(color='#f97316', **line_style)), row=2, col=1)
                fig.add_trace(go.Scatter(x=hours_arr, y=ts_h, name="水深", line=dict(color='#22c55e', **line_style), fill='tozeroy'), row=3, col=1)
                
                # 添加管顶红线
                fig.add_hline(y=info['Diameter'], line_dash="dash", line_color="red", row=3, col=1)
                
                fig.update_layout(height=500, margin=dict(t=20, b=0, l=0, r=0), showlegend=False, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(pd.DataFrame({'Q': ts_Q, 'v': ts_v, 'h': ts_h}))
        except Exception as e:
            st.error(f"读取结果出错: {e}")
    else:
        st.info("⚠️ 暂无水力结果，请先运行模拟。")

# ==========================================
# 5. 界面主逻辑
# ==========================================

st.title("🌊 城市雨水管网水力分析系统")

with st.sidebar:
    st.header("1. 数据导入")
    uploaded_file = st.file_uploader("上传管网数据", type=['xlsx', 'csv'])
    st.header("2. 模拟参数")
    sim_hours = st.slider("模拟时长 (小时)", 12, 48, 24)
    default_n = st.number_input("默认曼宁系数 (n)", 0.010, 0.020, 0.013, format="%.3f", step=0.001)

    if uploaded_file:
        df_pipe, error_msg, has_coords = load_and_process_data(uploaded_file)
        if error_msg:
            st.error(error_msg)
        else:
            if 'Manning' not in df_pipe.columns:
                df_pipe['Manning'] = default_n
            
            G, cycles_removed = build_graph(df_pipe)
            if cycles_removed > 0:
                st.warning(f"⚠️ 检测到管网中存在环路，已自动断开 {cycles_removed} 处连接。")
            
            sim_params_changed = (
                st.session_state['simulation_params'].get('hours') != sim_hours or
                st.session_state['simulation_params'].get('n') != default_n
            )
            
            if not st.session_state['has_results'] or sim_params_changed:
                if st.button("🚀 开始模拟计算", type="primary", use_container_width=True):
                    results = run_simulation_logic(G, df_pipe, sim_hours)
                    if results:
                        st.session_state['res_Q'] = results['res_Q']
                        st.session_state['res_v'] = results['res_v']
                        st.session_state['res_h'] = results['res_h']
                        st.session_state['all_pipe_ids'] = df_pipe['PipeID'].values
                        st.session_state['has_results'] = True
                        st.session_state['simulation_params'] = {'hours': sim_hours, 'n': default_n}
                        st.rerun()
            else:
                if st.button("🔄 重新计算", use_container_width=True):
                    st.session_state['has_results'] = False
                    st.rerun()

if uploaded_file and not error_msg:
    # 使用列布局：左侧地图，右侧详情
    col_map_area, col_details_area = st.columns([7, 3])
    
    with col_map_area:
        # --- 地图控制栏 ---
        c_ctrl1, c_ctrl2, c_spacer = st.columns([2, 2, 6])
        show_arrows = c_ctrl1.toggle("显示流向箭头", value=False)
        show_nodes = c_ctrl2.toggle("显示进水节点", value=False)
        
        df_map = df_pipe.copy()
        if has_coords:
            # 坐标转换
            df_map, trans_status = convert_coordinates(df_map)
            df_map = df_map.reset_index(drop=True) 
            
            if trans_status == "HK80":
                x_us, y_us, x_ds, y_ds = 'US_X_WGS84', 'US_Y_WGS84', 'DS_X_WGS84', 'DS_Y_WGS84'
            else:
                x_us, y_us, x_ds, y_ds = 'US_X', 'US_Y', 'DS_X', 'DS_Y'

            # --- 样式计算 & 高亮逻辑 ---
            d_min, d_max = df_map['Diameter'].min(), df_map['Diameter'].max()
            current_selected = st.session_state.get('selected_pipe_id')

            def get_style(row):
                # 如果是当前选中的管道，返回高亮色（品红）
                if str(row['PipeID']) == str(current_selected):
                    return pd.Series([[255, 0, 255, 255], max(5, row['Diameter'] * 8)])
                
                # 否则返回默认颜色（基于管径的蓝绿色系）
                d = row['Diameter']
                if d_max == d_min: ratio = 0.5
                else: ratio = (d - d_min) / (d_max - d_min)
                color = [int(0 + 100*ratio), int(100 + 155*ratio), 255, 200]
                width = max(2, d * 5)
                return pd.Series([color, width])

            df_map[['color', 'width']] = df_map.apply(get_style, axis=1)

            # --- 箭头几何计算 (PolygonLayer) ---
            def get_arrow_polygon(row):
                sx, sy = row[x_us], row[y_us]
                ex, ey = row[x_ds], row[y_ds]
                mx, my = (sx + ex) / 2, (sy + ey) / 2
                dx = ex - sx
                dy = ey - sy
                length = math.sqrt(dx*dx + dy*dy)
                if length == 0: return []
                ux, uy = dx/length, dy/length
                vx, vy = -uy, ux
                
                # 调整后的箭头尺寸
                scale = 0.00006 
                
                p1 = [mx + ux * scale * 1.5, my + uy * scale * 1.5]
                p2 = [mx - ux * scale + vx * scale * 0.8, my - uy * scale + vy * scale * 0.8]
                p3 = [mx - ux * scale - vx * scale * 0.8, my - uy * scale - vy * scale * 0.8]
                return [p1, p2, p3]

            df_map['arrow_polygon'] = df_map.apply(get_arrow_polygon, axis=1)

            mid_lat = (df_map[y_us].mean() + df_map[y_ds].mean()) / 2
            mid_lon = (df_map[x_us].mean() + df_map[x_ds].mean()) / 2
            view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=13, pitch=0)

            layers_list = []

            # 1. 管道线条层 (始终显示，可点击)
            line_layer = pdk.Layer(
                "LineLayer",
                df_map,
                get_source_position=[x_us, y_us],
                get_target_position=[x_ds, y_ds],
                get_color="color",
                get_width="width",
                width_min_pixels=2,
                pickable=True, # 关键：允许点击
                auto_highlight=True,
                highlight_color=[255, 255, 0, 255],
            )
            layers_list.append(line_layer)

            # 2. 箭头层 (根据开关显示，可点击)
            if show_arrows:
                arrow_layer = pdk.Layer(
                    "PolygonLayer",
                    df_map,
                    get_polygon="arrow_polygon",
                    get_fill_color=[255, 255, 255, 200],
                    pickable=True, # 关键：允许点击箭头选中管道
                    stroked=False,
                    auto_highlight=True,
                    highlight_color=[255, 0, 255, 255],
                )
                layers_list.append(arrow_layer)
            
            # 3. 节点层 (根据开关显示，不可点击)
            if show_nodes:
                nodes_df = df_map.groupby('UpstreamNode').agg({
                    x_us: 'first',
                    y_us: 'first'
                }).reset_index()
                
                node_layer = pdk.Layer(
                    "ScatterplotLayer",
                    nodes_df,
                    get_position=[x_us, y_us],
                    get_radius=10, 
                    get_fill_color=[255, 0, 0, 200], 
                    pickable=False, # 关键：禁止点击节点，防止索引混淆
                )
                layers_list.append(node_layer)

            deck = pdk.Deck(
                layers=layers_list,
                initial_view_state=view_state,
                map_style='mapbox://styles/mapbox/dark-v10',
                tooltip={
                    "html": "<b>ID:</b> {PipeID}<br/><b>管径:</b> {Diameter}m"
                }
            )
            
            # 渲染地图
            event = st.pydeck_chart(
                deck, 
                on_select="rerun", 
                selection_mode="single-object",
                use_container_width=True,
                height=600
            )
            
            # --- 核心修复：更健壮的选择逻辑 ---
            if event.selection:
                # 获取被点击对象的索引
                indices = event.selection.get("indices")
                
                # 只有当索引存在，且我们确定点击的是 line 或 arrow 层（它们共享 df_map）时才更新
                # 由于 node 层 pickable=False，所以这里的 index 一定对应 df_map
                if indices:
                    try:
                        idx = indices[0]
                        if idx < len(df_map):
                            selected_id = str(df_map.iloc[idx]['PipeID'])
                            # 只有当 ID 真正改变时才更新，避免不必要的刷新
                            if st.session_state['selected_pipe_id'] != selected_id:
                                st.session_state['selected_pipe_id'] = selected_id
                                st.rerun() # 强制刷新以更新右侧详情和地图高亮
                    except Exception as e:
                        print(f"Selection Error: {e}")
                        pass

        else:
            st.info("无坐标数据，无法显示地图。")

    with col_details_area:
        st.subheader("📋 详细信息")
        if st.session_state['selected_pipe_id']:
            render_pipe_details(st.session_state['selected_pipe_id'], df_pipe, sim_hours)
        else:
            st.info("👈 请在左侧地图中点击任意管道查看详情。")
            st.markdown("""
            **操作提示：**
            1. 缩放地图查看管网细节。
            2. 点击 **显示流向箭头** 查看水流方向。
            3. 点击 **显示进水节点** 查看进水口位置。
            4. 选中管道（变粉色）后，此处将显示水力模拟结果。
            """)

else:
    if not uploaded_file:
        st.info("👈 请先在左侧侧边栏上传数据文件。")
