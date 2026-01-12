import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import warnings
import plotly.graph_objects as go 

# ==========================================
# 0. 配置与初始化
# ==========================================
st.set_page_config(page_title="Urban Sewer Simulation (Custom Inflow)", layout="wide")
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
h1 { color: #2c3e50; }
.stPlotlyChart { border: 1px solid #e0e0e0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
div[data-testid="stMetricValue"] { font-size: 1.2rem; color: #2980b9; }
</style>
""", unsafe_allow_html=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. 核心计算类 (水力与水质)
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
        
        if np.any(mask_solve):
            theta_active = theta[mask_solve]
            D_active = D[mask_solve]
            K_t_active = K_target[mask_solve]
            coef_active = (D_active**2) / 8
            
            for _ in range(5):
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

class ASMKinetics(nn.Module):
    def __init__(self):
        super().__init__()
        self.uHO2 = 4.0; self.Ksw = 1.0; self.KO = 0.5; self.Yhw = 0.55
        self.qm = 0.5; self.XHf = 10.0; self.Kso4 = 62.85
        self.SO_sat = 8.0; self.Temp = 25.0; self.aw = 1.07
        
    def compute_rates(self, C, hydraulic_state):
        C = torch.clamp(C, min=0.0)
        XHw = C[:, 0:1]; Xs1 = C[:, 1:2]; SO = C[:, 3:4]; SF = C[:, 4:5]
        SHS = C[:, 6:7]; SSO4 = C[:, 7:8]

        vel = hydraulic_state['v']
        depth = hydraulic_state['h']
        
        depth_safe = torch.clamp(depth, min=1e-3)
        vel_safe = torch.clamp(vel, min=1e-3)
        
        K2_day = 3.93 * (vel_safe**0.5) / (depth_safe**1.5)
        Kla = K2_day / 24.0 * (1.024 ** (self.Temp - 20))
        Kla = torch.clamp(Kla, max=100.0)
        phi = self.aw ** (self.Temp - 20)
        
        M_SF = SF / (self.Ksw + SF + 1e-6)
        M_SO = SO / (self.KO + SO + 1e-6)
        M_SO_lim = self.KO / (self.KO + SO + 1e-6)
        M_SSO4 = SSO4 / (self.Kso4 + SSO4 + 1e-6)

        rho_grw = self.uHO2 * M_SF * M_SO * XHw * phi
        rho_srb = 0.05 * M_SF * M_SSO4 * self.XHf * M_SO_lim * phi
        rho_sox = 2.0 * M_SO * SHS * phi
        rho_hyd = 2.0 * Xs1 * (XHw / (XHw + Xs1 + 1e-6)) * M_SO * phi

        dXHw = rho_grw - 0.1 * XHw
        dXs1 = -rho_hyd
        dXs2 = torch.zeros_like(Xs1)
        dSO  = Kla * (self.SO_sat - SO) - ((1-self.Yhw)/self.Yhw)*rho_grw - 2.0*rho_sox
        dSF  = rho_hyd - (1/self.Yhw)*rho_grw - rho_srb
        dSac = torch.zeros_like(SF)
        dSHS = rho_srb - rho_sox
        dSSO4= -rho_srb + rho_sox
        dCH4 = 0.1 * rho_srb
        dSprop = torch.zeros_like(SF); dH2 = torch.zeros_like(SF)

        return torch.cat([dXHw, dXs1, dXs2, dSO, dSF, dSac, dSHS, dSSO4, dCH4, dSprop, dH2], dim=1)

# ==========================================
# 2. 数据处理与模拟逻辑
# ==========================================

@st.cache_data
def process_uploaded_data(df):
    # 1. 标准化列名映射
    col_map = {
        'name': 'PipeID', 'start': 'UpstreamNode', 'end': 'DownstreamNode',
        'length': 'Length', 'diameter': 'Diameter', 'slope': 'Slope',
        'us_x': 'US_X', 'us_y': 'US_Y', 'ds_x': 'DS_X', 'ds_y': 'DS_Y',
        # 允许用户使用不同的变体，但统一映射到 'inflow_baseline'
        'inflow': 'inflow_baseline', 
        'flow': 'inflow_baseline',
        'base_flow': 'inflow_baseline',
        'q_base': 'inflow_baseline'
    }
    df = df.rename(columns=col_map)
    
    # 2. 检查必要列
    required = ['PipeID', 'UpstreamNode', 'DownstreamNode', 'Length', 'Diameter', 'Slope']
    if any(c not in df.columns for c in required): 
        return None, f"Missing required columns. Found: {list(df.columns)}"
    
    # 3. 检查 inflow_baseline
    if 'inflow_baseline' not in df.columns:
        return None, "Missing required column: 'inflow_baseline' (or 'inflow', 'flow', 'base_flow'). Unit should be m³/s."
    
    # 4. 数据清洗
    df['UpstreamNode'] = df['UpstreamNode'].astype(str)
    df['DownstreamNode'] = df['DownstreamNode'].astype(str)
    df['Slope'] = df['Slope'].clip(lower=0.001)
    df['inflow_baseline'] = df['inflow_baseline'].fillna(0.0) # 缺失值补0
    
    if 'Manning' not in df.columns: df['Manning'] = 0.013
    
    if 'US_X' in df.columns and 'DS_X' in df.columns:
        df['Mid_X'] = (df['US_X'] + df['DS_X']) / 2
        df['Mid_Y'] = (df['US_Y'] + df['DS_Y']) / 2
        
    return df, None

@st.cache_data
def build_graph(df_pipe):
    G = nx.DiGraph()
    for _, row in df_pipe.iterrows():
        G.add_edge(row['UpstreamNode'], row['DownstreamNode'], pipe_id=row['PipeID'], length=row['Length'])
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G)
            G.remove_edge(*cycle[0])
        except: break
    return G

@st.cache_data
def run_hydraulic_simulation(df_pipe, sim_hours):
    G = build_graph(df_pipe)
    topo_nodes = list(nx.topological_sort(G))
    all_nodes = list(G.nodes())
    
    np.random.seed(42)
    node_inflows = {}
    time_steps = np.arange(sim_hours)
    
    # 24小时循环模式
    hour_of_day = time_steps % 24
    
    # === 修改点：从 DataFrame 中聚合用户提供的 inflow_baseline ===
    # 逻辑：将 CSV 中每一行的 inflow_baseline 归属到该行的 UpstreamNode
    # 如果有多个管道从同一个节点出发，我们假设这些 inflow 是累加的外部流入
    node_baseline_map = df_pipe.groupby('UpstreamNode')['inflow_baseline'].sum().to_dict()
    
    for node in all_nodes:
        # 获取用户定义的基准流量，如果节点没有作为 UpstreamNode 出现（例如纯末端节点），则默认为 0
        base = node_baseline_map.get(node, 0.0)
        
        # 仍然应用日变化模式 (Diurnal Pattern)
        # 如果 base 为 0，则整个序列为 0
        pat = 0.3 + 0.6 * np.exp(-((hour_of_day - 8)**2) / 8) + 0.5 * np.exp(-((hour_of_day - 20)**2) / 8)
        
        # 确保最小流量不为负，且给一个极小值防止除零（如果 base > 0）
        if base > 0:
            node_inflows[node] = np.maximum(base * pat, 0.0001)
        else:
            node_inflows[node] = np.zeros(sim_hours)

    solver = VectorizedHydraulics()
    num_pipes = len(df_pipe)
    res_Q = np.zeros((num_pipes, sim_hours))
    res_v = np.zeros((num_pipes, sim_hours))
    res_h = np.zeros((num_pipes, sim_hours))
    pipe_id_to_idx = {pid: i for i, pid in enumerate(df_pipe['PipeID'])}
    
    for t in range(sim_hours):
        node_acc = {n: node_inflows[n][t] for n in all_nodes}
        current_Q_map = {}
        for u in topo_nodes:
            total_in = node_acc[u]
            out_edges = list(G.out_edges(u, data=True))
            
            # 如果没有出边，流量流出系统（到达WWTP或Outfall）
            if not out_edges: continue
            
            # 简单假设：均分给下游管道
            flow_per = total_in / len(out_edges)
            for _, v_node, data in out_edges:
                pid = data['pipe_id']
                current_Q_map[pid] = flow_per
                if v_node in node_acc: node_acc[v_node] += flow_per
        
        curr_Q_arr = np.zeros(num_pipes)
        for pid, q_val in current_Q_map.items():
            if pid in pipe_id_to_idx:
                curr_Q_arr[pipe_id_to_idx[pid]] = q_val
                
        h, v = solver.solve_normal_depth(
            curr_Q_arr, df_pipe['Diameter'].values, df_pipe['Slope'].values, df_pipe['Manning'].values
        )
        res_Q[:, t] = curr_Q_arr
        res_v[:, t] = v
        res_h[:, t] = h
        
    return {'Q': res_Q, 'v': res_v, 'h': res_h}

@st.cache_data
def run_wq_simulation(df_pipe, hyd_res_dict, use_seawater, use_food_waste):
    Q = hyd_res_dict['Q']; v = hyd_res_dict['v']; h = hyd_res_dict['h']
    sim_steps = Q.shape[1]
    
    nodes_uniq = sorted(list(set(df_pipe['UpstreamNode']).union(set(df_pipe['DownstreamNode']))))
    n_map = {n: i for i, n in enumerate(nodes_uniq)}
    edge_src = [n_map[u] for u in df_pipe['UpstreamNode']]
    edge_dst = [n_map[v] for v in df_pipe['DownstreamNode']]
    
    edge_idx = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
    hyd_data = {
        'Q': torch.tensor(Q.T, dtype=torch.float32, device=device),
        'v': torch.tensor(v.T, dtype=torch.float32, device=device),
        'h': torch.tensor(h.T, dtype=torch.float32, device=device),
        'L': torch.tensor(df_pipe['Length'].values, dtype=torch.float32, device=device).unsqueeze(0).expand(sim_steps, -1)
    }
    
    num_nodes = len(nodes_uniq)
    C_nodes = torch.zeros((num_nodes, 11), device=device) + 1e-6
    C_nodes[:, 3] = 6.0 
    
    asm = ASMKinetics().to(device)
    history_pipes = []
    
    # 识别源头节点（入度为0的节点），只有这些节点会持续补充污染物
    # 注意：这里我们简单假设所有源头都有污染物输入
    # 如果想更精确，可以结合 inflow_baseline > 0 的节点来判断
    G = build_graph(df_pipe)
    in_degs = [G.in_degree(n) for n in nodes_uniq]
    src_idxs = torch.tensor([i for i, d in enumerate(in_degs) if d == 0], dtype=torch.long, device=device)
    
    so4_baseline = 120.0 if use_seawater else 20.0
    cod_multiplier = 2.0 if use_food_waste else 1.0
    
    for t in range(sim_steps):
        if len(src_idxs) > 0:
            hour_of_day = t % 24
            pattern = 1.0 + 0.5 * np.sin(2*np.pi*(hour_of_day-8)/24)
            
            C_nodes[src_idxs, 0] = 30.0 * pattern * cod_multiplier 
            C_nodes[src_idxs, 1] = 150.0 * pattern * cod_multiplier 
            C_nodes[src_idxs, 4] = 100.0 * pattern * cod_multiplier 
            C_nodes[src_idxs, 7] = so4_baseline 
        
        curr_v = hyd_data['v'][t]; curr_L = hyd_data['L'][t]; curr_Q = hyd_data['Q'][t]
        res_time = torch.clamp((curr_L / (curr_v + 1e-4)) / 3600.0, max=1.0)
        
        C_in = C_nodes[edge_idx[0]]
        hyd_state_t = {'v': curr_v.unsqueeze(1), 'h': hyd_data['h'][t].unsqueeze(1)}
        
        rates = asm.compute_rates(C_in, hyd_state_t)
        C_out = C_in + rates * res_time.unsqueeze(1)
        C_out = torch.clamp(C_out, min=1e-6)
        
        history_pipes.append(C_out.clone().cpu())
        
        mass = C_out * curr_Q.unsqueeze(1)
        tot_m = torch.zeros((num_nodes, 11), device=device)
        tot_q = torch.zeros((num_nodes, 1), device=device)
        tot_m.index_add_(0, edge_idx[1], mass)
        tot_q.index_add_(0, edge_idx[1], curr_Q.unsqueeze(1))
        
        mask = (tot_q > 1e-6).squeeze()
        valid_dst = torch.unique(edge_idx[1])
        valid_dst = valid_dst[mask[valid_dst]]
        if len(valid_dst) > 0:
            C_nodes[valid_dst] = tot_m[valid_dst] / tot_q[valid_dst]
            
    return torch.stack(history_pipes, dim=0).numpy()

# ==========================================
# 3. HRT 计算功能
# ==========================================

def calculate_downstream_hrt(start_node, G, df_pipe, avg_velocities):
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
    max_hrt = 0
    pipe_v_map = dict(zip(df_pipe['PipeID'], avg_velocities))
    
    for sink in sinks:
        try:
            paths = list(nx.all_simple_paths(G, source=start_node, target=sink))
            for path in paths:
                path_hrt = 0
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    edge_data = G.get_edge_data(u, v)
                    pid = edge_data['pipe_id']
                    length = edge_data['length']
                    vel = max(pipe_v_map.get(pid, 0.1), 0.01) 
                    path_hrt += (length / vel) / 3600.0
                if path_hrt > max_hrt:
                    max_hrt = path_hrt
        except nx.NetworkXNoPath:
            continue
    return max_hrt

# ==========================================
# 4. 绘图辅助函数
# ==========================================

def create_interactive_map(df_pipe):
    fig = go.Figure()

    x_lines = []
    y_lines = []
    for _, row in df_pipe.iterrows():
        x_lines.extend([row['US_X'], row['DS_X'], None])
        y_lines.extend([row['US_Y'], row['DS_Y'], None])
    
    fig.add_trace(go.Scatter(
        x=x_lines, y=y_lines,
        mode='lines',
        line=dict(color='#bdc3c7', width=2),
        hoverinfo='skip',
        name='Pipes'
    ))

    # 根据 inflow_baseline 大小调整节点颜色或大小（可选）
    # 这里简单展示所有节点
    fig.add_trace(go.Scatter(
        x=df_pipe['Mid_X'], y=df_pipe['Mid_Y'],
        mode='markers',
        marker=dict(size=8, color='rgba(231, 76, 60, 0.7)', line=dict(width=1, color='white')),
        name='Select Pipe',
        text=df_pipe['PipeID'],
        hovertemplate='<b>Pipe: %{text}</b><br>Inflow Base: %{customdata[0]:.4f} m³/s<extra></extra>',
        customdata=np.stack((df_pipe['inflow_baseline'], df_pipe.index), axis=-1)
    ))

    us_nodes = set(df_pipe['UpstreamNode'])
    ds_nodes = set(df_pipe['DownstreamNode'])
    sinks = ds_nodes - us_nodes
    
    sink_x = []
    sink_y = []
    for sink in sinks:
        pipe_ending = df_pipe[df_pipe['DownstreamNode'] == sink].iloc[0]
        sink_x.append(pipe_ending['DS_X'])
        sink_y.append(pipe_ending['DS_Y'])

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
        title="Network Map",
        xaxis_title="X (m)", yaxis_title="Y (m)",
        showlegend=True,
        hovermode='closest',
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        dragmode='pan',
        plot_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ==========================================
# 5. Streamlit 界面
# ==========================================

st.title("🏙️ Urban Drainage Network (Custom Inflow & HRT)")

with st.sidebar:
    st.header("1. Data Import")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], 
                                     help="Must contain columns: PipeID, UpstreamNode, DownstreamNode, Length, Diameter, Slope, inflow_baseline (m³/s)")
    
    st.header("2. Simulation Control")
    sim_hours = st.slider("Duration (Hours)", min_value=24, max_value=168, value=48, step=12)
    
    st.divider()
    st.header("3. Scenario Settings")
    use_seawater = st.toggle("🌊 Seawater Flushing", value=False)
    use_food_waste = st.toggle("🍔 Food Waste Disposer", value=False)
    
    if uploaded_file:
        st.divider()
        st.info("Using user-provided 'inflow_baseline' for simulation.")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    df_pipe, error_msg = process_uploaded_data(df_raw)
    
    if error_msg:
        st.error(error_msg)
    elif df_pipe is not None:
        with st.spinner("Processing Hydraulics..."):
            hyd_results = run_hydraulic_simulation(df_pipe, sim_hours)
        
        with st.spinner("Processing Water Quality..."):
            wq_results = run_wq_simulation(df_pipe, hyd_results, use_seawater, use_food_waste)
            
        st.success(f"Simulation Complete! Used inflow data from CSV.")

        col_map, col_detail = st.columns([3, 2])
        G_network = build_graph(df_pipe)
        
        with col_map:
            st.subheader("🗺️ Network Map")
            if 'US_X' in df_pipe.columns:
                fig = create_interactive_map(df_pipe)
                selection = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
                
                selected_pipe_idx = None
                if selection and selection['selection']['points']:
                    for point in selection['selection']['points']:
                        # customdata is now [inflow, index]
                        if 'customdata' in point:
                            selected_pipe_idx = point['customdata'][1] 
                            break
            else:
                st.warning("No coordinate data found in CSV.")

        with col_detail:
            st.subheader("📊 Results Inspector")
            
            if selected_pipe_idx is not None:
                try:
                    idx = int(selected_pipe_idx)
                    pipe_info = df_pipe.iloc[idx]
                    
                    # HRT Calculation
                    avg_velocities = np.mean(hyd_results['v'], axis=1)
                    start_node = str(pipe_info['DownstreamNode'])
                    current_pipe_vel = max(avg_velocities[idx], 0.01)
                    current_pipe_hrt = (pipe_info['Length'] / current_pipe_vel) / 3600.0
                    downstream_hrt = calculate_downstream_hrt(start_node, G_network, df_pipe, avg_velocities)
                    total_hrt = current_pipe_hrt + downstream_hrt
                    
                    st.markdown(f"### Pipe: `{pipe_info['PipeID']}`")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Length", f"{pipe_info['Length']:.1f} m")
                    m2.metric("Inflow Base", f"{pipe_info['inflow_baseline']:.4f}", "m³/s")
                    m3.metric("⏱️ HRT to WWTP", f"{total_hrt:.2f} h")
                    
                    st.divider()

                    tab1, tab2 = st.tabs(["💧 Hydraulics", "🧪 Water Quality"])
                    ts = range(sim_hours)
                    
                    with tab1:
                        fig_h, ax_h = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
                        ax_h[0].plot(ts, hyd_results['Q'][idx], 'b-', lw=2)
                        ax_h[0].set_title("Flow Rate (Q)", fontsize=10)
                        ax_h[0].set_ylabel("m³/s")
                        ax_h[0].grid(True, alpha=0.3)
                        
                        ax_h[1].plot(ts, hyd_results['h'][idx], 'g-', lw=2)
                        ax_h[1].axhline(pipe_info['Diameter'], color='r', ls=':', label='Max')
                        ax_h[1].set_title("Water Depth (h)", fontsize=10)
                        ax_h[1].set_ylabel("m")
                        ax_h[1].set_xlabel("Time (h)")
                        ax_h[1].grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_h)

                    with tab2:
                        cod_series = wq_results[:, idx, 1] + wq_results[:, idx, 4] 
                        do_series = wq_results[:, idx, 3]  
                        so4_series = wq_results[:, idx, 7] 
                        h2s_series = wq_results[:, idx, 6] 
                        ch4_series = wq_results[:, idx, 8] 
                        
                        fig_w, ax_w = plt.subplots(5, 1, figsize=(6, 12), sharex=True)
                        ax_w[0].plot(ts, cod_series, color='#8e44ad', lw=2)
                        ax_w[0].set_title("Total COD (mg/L)", fontsize=10, loc='left')
                        ax_w[0].grid(True, alpha=0.3)
                        
                        ax_w[1].plot(ts, do_series, color='#3498db', lw=2)
                        ax_w[1].set_title("Dissolved Oxygen (DO) (mg/L)", fontsize=10, loc='left')
                        ax_w[1].grid(True, alpha=0.3)
                        
                        ax_w[2].plot(ts, so4_series, color='#f39c12', lw=2)
                        ax_w[2].set_title("Sulfate (SO₄²⁻) (mgS/L)", fontsize=10, loc='left')
                        ax_w[2].grid(True, alpha=0.3)
                        
                        ax_w[3].plot(ts, h2s_series, color='#e74c3c', lw=2)
                        ax_w[3].set_title("Sulfide (H₂S) (mgS/L)", fontsize=10, loc='left')
                        ax_w[3].grid(True, alpha=0.3)
                        
                        ax_w[4].plot(ts, ch4_series, color='#d35400', lw=2, linestyle='--')
                        ax_w[4].set_title("Methane (CH₄) (mg/L)", fontsize=10, loc='left')
                        ax_w[4].set_xlabel("Time (h)")
                        ax_w[4].grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_w)
                        
                except Exception as e:
                    st.error(f"Error displaying data: {e}")
            else:
                st.info("Select a red node on the map to view HRT and time-series data.")

else:
    st.info("👈 Upload your network CSV to begin. Ensure it has an 'inflow_baseline' column.")
