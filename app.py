import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ==========================================
# 1. 模拟数据生成 (如果你有自己的CSV读取逻辑，请替换此部分)
# ==========================================
def generate_dummy_data():
    # 创建一个简单的 Y 型网络
    data = {
        'PipeID': ['P01', 'P02', 'P03', 'P04'],
        'UpstreamNode': ['N1', 'N2', 'N3', 'N4'],
        'DownstreamNode': ['N3', 'N3', 'N4', 'Out1'],
        'Length': [100, 120, 80, 200],
        'Diameter': [0.5, 0.5, 0.8, 1.0],
        'Slope': [0.005, 0.005, 0.002, 0.001],
        'Manning': [0.013, 0.013, 0.013, 0.013],
        'inflow_baseline': [0.05, 0.04, 0.0, 0.0],
        # 坐标用于绘图
        'US_X': [0, 0, 100, 200],
        'US_Y': [100, 0, 50, 50],
        'DS_X': [100, 100, 200, 300],
        'DS_Y': [50, 50, 50, 50]
    }
    df = pd.DataFrame(data)
    # 计算中点用于放置红点
    df['Mid_X'] = (df['US_X'] + df['DS_X']) / 2
    df['Mid_Y'] = (df['US_Y'] + df['DS_Y']) / 2
    return df

# ==========================================
# 2. 模拟计算核心 (简化版)
# ==========================================
def build_network_graph(df_pipe):
    G = nx.DiGraph()
    for _, row in df_pipe.iterrows():
        G.add_edge(str(row['UpstreamNode']), str(row['DownstreamNode']), 
                   weight=row['Length'], pipe_id=row['PipeID'])
    return G

def simulate_hydraulics(df_pipe, hours=24):
    steps = hours
    n_pipes = len(df_pipe)
    # 模拟结果：Q (流量), v (流速), h (水深)
    # 随机生成一些波动数据
    Q = np.random.uniform(0.02, 0.1, (n_pipes, steps))
    v = np.random.uniform(0.5, 1.5, (n_pipes, steps))
    h = np.random.uniform(0.1, 0.4, (n_pipes, steps))
    return {'Q': Q, 'v': v, 'h': h}

def simulate_wq(hyd_results, df_pipe, hours=24):
    # 模拟水质: 9个参数 (COD, DO, etc.)
    # 形状: (时间, 管道数, 参数数)
    return np.random.uniform(0, 10, (hours, len(df_pipe), 9))

def calculate_downstream_hrt(start_node, G, df_pipe, avg_velocities):
    # 简化的下游 HRT 计算
    total_time = 0.0
    current_node = start_node
    
    # 简单的遍历直到没有下游
    while True:
        successors = list(G.successors(current_node))
        if not successors:
            break
        next_node = successors[0] # 假设无分叉
        
        # 找到连接这两个节点的管道
        edge_data = G.get_edge_data(current_node, next_node)
        # 这里为了简化，需要反查管道索引，实际项目中建议用字典映射
        pipe_row = df_pipe[df_pipe['PipeID'] == edge_data['pipe_id']]
        if not pipe_row.empty:
            idx = pipe_row.index[0]
            length = pipe_row.iloc[0]['Length']
            vel = max(avg_velocities[idx], 0.01)
            total_time += (length / vel) / 3600.0
        
        current_node = next_node
        
    return total_time

# ==========================================
# 3. 绘图函数 (关键修改在 Trace 1)
# ==========================================
def create_interactive_map(df_pipe):
    fig = go.Figure()

    # --- Trace 0: 灰色管道连线 (Lines) ---
    # curveNumber = 0
    x_lines = []
    y_lines = []
    for _, row in df_pipe.iterrows():
        x_lines.extend([row['US_X'], row['DS_X'], None])
        y_lines.extend([row['US_Y'], row['DS_Y'], None])
    
    fig.add_trace(go.Scatter(
        x=x_lines, y=y_lines,
        mode='lines',
        line=dict(color='#bdc3c7', width=2),
        hoverinfo='skip', # 禁用悬停，避免干扰
        name='Pipes'
    ))

    # --- Trace 1: 红色交互点 (Red Dots) ---
    # curveNumber = 1
    # 我们不需要在这里传 customdata 用于索引，因为我们将使用 pointIndex
    fig.add_trace(go.Scatter(
        x=df_pipe['Mid_X'], y=df_pipe['Mid_Y'],
        mode='markers',
        marker=dict(size=10, color='rgba(231, 76, 60, 0.9)', line=dict(width=1, color='white')),
        name='Select Pipe',
        text=df_pipe['PipeID'],
        hovertemplate='<b>Pipe: %{text}</b><br>Click to view details<extra></extra>'
    ))

    # --- Trace 2: 绿色终点/污水厂 (WWTP) ---
    # curveNumber = 2
    us_nodes = set(df_pipe['UpstreamNode'])
    ds_nodes = set(df_pipe['DownstreamNode'])
    sinks = ds_nodes - us_nodes
    
    sink_x = []
    sink_y = []
    for sink in sinks:
        pipe_ending = df_pipe[df_pipe['DownstreamNode'] == sink]
        if not pipe_ending.empty:
            sink_x.append(pipe_ending.iloc[0]['DS_X'])
            sink_y.append(pipe_ending.iloc[0]['DS_Y'])

    if sink_x:
        fig.add_trace(go.Scatter(
            x=sink_x, y=sink_y,
            mode='markers',
            marker=dict(size=15, color='#2ecc71', symbol='square', line=dict(width=2, color='white')),
            name='WWTP / Outfall',
            hoverinfo='text',
            text=['WWTP / Outfall'] * len(sink_x)
        ))

    fig.update_layout(
        title="Network Map (Click Red Nodes)",
        xaxis_title="X (m)", yaxis_title="Y (m)",
        showlegend=True,
        hovermode='closest',
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
        dragmode='pan',
        plot_bgcolor='white'
    )
    return fig

# ==========================================
# 4. Streamlit 主程序
# ==========================================
def main():
    st.set_page_config(page_title="Sewer Digital Twin", layout="wide")
    st.title("💧 Sewer Digital Twin: Hydraulic & WQ Analysis")

    # --- 数据加载 ---
    # 实际使用时：
    # uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    # if uploaded_file:
    #     df_pipe = pd.read_csv(uploaded_file)
    # else:
    #     df_pipe = generate_dummy_data() # fallback
    
    # 演示用：直接生成数据
    df_pipe = generate_dummy_data()
    
    if df_pipe is not None:
        # 运行模拟
        G_network = build_network_graph(df_pipe)
        sim_hours = 24
        hyd_results = simulate_hydraulics(df_pipe, sim_hours)
        wq_results = simulate_wq(hyd_results, df_pipe, sim_hours)

        # 布局
        col_map, col_detail = st.columns([1.2, 1])

        # --- 地图交互部分 ---
        with col_map:
            st.subheader("🗺️ Network Map")
            fig = create_interactive_map(df_pipe)
            
            # 渲染图表并开启选择事件
            selection = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
            
            selected_pipe_idx = None
            
            # ======================================================
            # 关键修复逻辑：解析点击事件
            # ======================================================
            if selection and "selection" in selection and "points" in selection["selection"]:
                points = selection["selection"]["points"]
                if points:
                    for point in points:
                        # 1. 检查 curveNumber (图层编号)
                        # Trace 0 = 线 (不处理)
                        # Trace 1 = 红点 (我们要处理的)
                        # Trace 2 = 绿方块 (不处理)
                        curve_num = point.get("curveNumber")
                        
                        if curve_num == 1:
                            # 2. 获取 pointIndex
                            # pointIndex 是该点在数据数组中的索引 (0, 1, 2...)
                            # 对应 df_pipe 的行号 (iloc)
                            idx = point.get("pointIndex")
                            
                            if idx is not None:
                                selected_pipe_idx = idx
                                # 找到一个有效点击就退出循环
                                break 
            # ======================================================

        # --- 结果详情部分 ---
        with col_detail:
            st.subheader("📊 Results Inspector")
            
            if selected_pipe_idx is not None:
                # 再次确认索引是否越界 (安全起见)
                if 0 <= selected_pipe_idx < len(df_pipe):
                    pipe_info = df_pipe.iloc[selected_pipe_idx]
                    
                    # 计算指标
                    avg_velocities = np.mean(hyd_results['v'], axis=1)
                    start_node = str(pipe_info['DownstreamNode'])
                    current_pipe_vel = max(avg_velocities[selected_pipe_idx], 0.01)
                    current_pipe_hrt = (pipe_info['Length'] / current_pipe_vel) / 3600.0
                    downstream_hrt = calculate_downstream_hrt(start_node, G_network, df_pipe, avg_velocities)
                    total_hrt = current_pipe_hrt + downstream_hrt
                    
                    # 显示头部信息
                    st.markdown(f"### Selected Pipe: `{pipe_info['PipeID']}`")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Length", f"{pipe_info['Length']:.1f} m")
                    m2.metric("Diameter", f"{pipe_info['Diameter']:.2f} m")
                    m3.metric("HRT to Outfall", f"{total_hrt:.2f} h")
                    
                    st.divider()

                    # 显示图表
                    tab1, tab2 = st.tabs(["💧 Hydraulics", "🧪 Water Quality"])
                    ts = range(sim_hours)
                    
                    with tab1:
                        fig_h, ax_h = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
                        ax_h[0].plot(ts, hyd_results['Q'][selected_pipe_idx], 'b-', lw=2)
                        ax_h[0].set_ylabel("Flow (m³/s)")
                        ax_h[0].grid(True, alpha=0.3)
                        
                        ax_h[1].plot(ts, hyd_results['h'][selected_pipe_idx], 'g-', lw=2)
                        ax_h[1].axhline(pipe_info['Diameter'], color='r', ls=':', label='Max')
                        ax_h[1].set_ylabel("Depth (m)")
                        ax_h[1].set_xlabel("Time (h)")
                        ax_h[1].grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_h)

                    with tab2:
                        # 示例：只画一个 COD
                        cod_series = wq_results[:, selected_pipe_idx, 0] # 假设第0个是COD
                        fig_w, ax_w = plt.subplots(figsize=(5, 3))
                        ax_w.plot(ts, cod_series, color='#8e44ad', lw=2)
                        ax_w.set_title("COD Concentration")
                        ax_w.set_ylabel("mg/L")
                        ax_w.set_xlabel("Time (h)")
                        ax_w.grid(True, alpha=0.3)
                        st.pyplot(fig_w)
                else:
                    st.error(f"Index error: {selected_pipe_idx} is out of bounds.")
            else:
                st.info("👈 Click on a **RED node** in the map to see details.")
                st.caption("Note: Clicking on gray lines or green squares will not trigger updates.")

if __name__ == "__main__":
    main()
