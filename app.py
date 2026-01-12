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
st.set_page_config(page_title="城市管网模拟 (交互版)", layout="wide")
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
h1 { color: #2c3e50; }
.stPlotlyChart { border: 1px solid #e0e0e0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. 核心计算类
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
        P_final = (D / 2) * theta
        v = np.zeros_like(Q_target)
        valid_A = A_final > 1e-6
        v[valid_A] = Q_target[valid_A] / A_final[valid_A]
        
        return h, v, A_final, P_final

class ASMKinetics(nn.Module):
    def __init__(self):
        super().__init__()
        self.uHO2 = 4.0; self.Ksw = 1.0; self.KO = 0.5; self.Yhw = 0.55
        self.qm = 0.5; self.dHana = 0.1; self.XHf = 10.0; self.Kso4 = 62.85
        self.SO_sat = 8.0; self.Temp = 25.0; self.aw = 1.07
        
    def compute_rates(self, C, hydraulic_state):
        C = torch.clamp(C, min=0.0)
        XHw, Xs1, Xs2, SO, SF, Sac, SHS, SSO4, CH4, Sprop, H2 = [C[:, i:i+1] for i in range(11)]

        vel = hydraulic_state['v']; depth = hydraulic_state['h']
        depth_safe = torch.clamp(depth, min=1e-3)
        vel_safe = torch.clamp(vel, min=1e-3)
        
        K2_day = 3.93 * (vel_safe**0.5) / (depth_safe**1.5)
        Kla = K2_day / 24.0
        Kla = Kla * (1.024 ** (self.Temp - 20))
        Kla = torch.clamp(Kla, max=100.0)
        phi = self.aw ** (self.Temp - 20)
        
        M_SF = SF / (self.Ksw + SF + 1e-6)
        M_SO = SO / (self.KO + SO + 1e-6)
        M_SO_lim = self.KO / (self.KO + SO + 1e-6)
        M_SSO4 = SSO4 / (self.Kso4 + SSO4 + 1e-6)

        rho_grw = self.uHO2 * M_SF * M_SO * XHw * phi
        rho_maint = self.qm * M_SO * XHw * phi
        rho_srb = 0.05 * M_SF * M_SSO4 * self.XHf * M_SO_lim * phi
        rho_sox = 2.0 * M_SO * SHS * phi
        rho_hyd = 2.0 * Xs1 * (XHw / (XHw + Xs1 + 1e-6)) * M_SO * phi

        dXHw = rho_grw - rho_maint
        dXs1 = -rho_hyd
        dXs2 = torch.zeros_like(Xs2) 
        dSO  = Kla * (self.SO_sat - SO) - ((1-self.Yhw)/self.Yhw)*rho_grw - rho_maint - 2.0*rho_sox
        dSF  = rho_hyd - (1/self.Yhw)*rho_grw - rho_srb
        dSac = torch.zeros_like(Sac); dSHS = rho_srb - rho_sox
        dSSO4= -rho_srb + rho_sox; dCH4 = 0.1 * rho_srb 
        dSprop = torch.zeros_like(Sprop); dH2 = torch.zeros_like(H2)

        return torch.cat([dXHw, dXs1, dXs2, dSO, dSF, dSac, dSHS, dSSO4, dCH4, dSprop, dH2], dim=1)

# ==========================================
# 2. 数据处理函数
# ==========================================

def process_uploaded_data(df):
    col_map = {
        'name': 'PipeID', 'start': 'UpstreamNode', 'end': 'DownstreamNode',
        'length': 'Length', 'diameter': 'Diameter', 'slope': 'Slope',
        'us_x': 'US_X', 'us_y': 'US_Y', 'ds_x': 'DS_X', 'ds_y': 'DS_Y'
    }
    df = df.rename(columns=col_map)
    required = ['PipeID', 'UpstreamNode', 'DownstreamNode', 'Length', 'Diameter', 'Slope']
    if any(c not in df.columns for c in required): return None
    
    df['UpstreamNode'] = df['UpstreamNode'].astype(str)
    df['DownstreamNode'] = df['DownstreamNode'].astype(str)
    if (df['Slope'] <= 0).any():
        df['Slope'] = df['Slope'].clip(lower=0.001)
    if 'Manning' not in df.columns: df['Manning'] = 0.013
    
    # 计算管段中心点，用于交互点击
    if 'US_X' in df.columns and 'DS_X' in df.columns:
        df['Mid_X'] = (df['US_X'] + df['DS_X']) / 2
        df['Mid_Y'] = (df['US_Y'] + df['DS_Y']) / 2
        
    return df

def generate_heterogeneous_inflows(nodes, hours=24):
    np.random.seed(42)
    node_inflows = {}
    time_steps = np.arange(hours)
    for node in nodes:
        base_flow = np.random.uniform(0.001, 0.008) 
        morning_peak = 7 + np.random.normal(0, 0.5) 
        evening_peak = 19 + np.random.normal(0, 0.5)
        pattern = 0.3 + 0.6 * np.exp(-((time_steps - morning_peak)**2) / 8) + \
                  0.5 * np.exp(-((time_steps - evening_peak)**2) / 8)
        pattern += np.random.normal(0, 0.02, size=hours)
        node_inflows[node] = np.maximum(base_flow * pattern, 0.0001)
    return node_inflows

# ==========================================
# 3. 绘图辅助函数
# ==========================================

def create_interactive_map(df_pipe):
    fig = go.Figure()

    # 1. 绘制管线 (背景层，不可点击或点击无反馈)
    x_lines = []
    y_lines = []
    for _, row in df_pipe.iterrows():
        x_lines.extend([row['US_X'], row['DS_X'], None])
        y_lines.extend([row['US_Y'], row['DS_Y'], None])
    
    fig.add_trace(go.Scatter(
        x=x_lines, y=y_lines,
        mode='lines',
        line=dict(color='gray', width=2),
        hoverinfo='skip', # 禁用悬停
        name='Pipes'
    ))

    # 2. 绘制节点 (简单的装饰)
    fig.add_trace(go.Scatter(
        x=df_pipe['US_X'], y=df_pipe['US_Y'],
        mode='markers',
        marker=dict(size=6, color='blue'),
        name='Nodes',
        hoverinfo='skip'
    ))

    # 3. 绘制管段中心交互点 (关键层)
    # 这是用户真正点击的对象
    fig.add_trace(go.Scatter(
        x=df_pipe['Mid_X'], y=df_pipe['Mid_Y'],
        mode='markers',
        marker=dict(size=10, color='rgba(255, 0, 0, 0.5)', line=dict(width=1, color='red')),
        name='Pipe Select',
        text=df_pipe['PipeID'], # 悬停显示 PipeID
        hovertemplate='<b>Pipe: %{text}</b><extra></extra>',
        # 这里的 customdata 非常重要，它将直接传递给回调
        customdata=df_pipe.index 
    ))

    fig.update_layout(
        title="管网交互地图 (点击红色锚点查看详情)",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        showlegend=False,
        hovermode='closest',
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
        # 锁定缩放比例，防止地图变形
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    return fig

# ==========================================
# 4. Streamlit 界面
# ==========================================

st.title("🏙️ 城市排水管网模拟系统 (交互版)")

# --- 侧边栏 ---
with st.sidebar:
    st.header("1. 数据导入")
    uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
    
    with st.expander("查看数据格式示例"):
        st.markdown("""
        CSV 必须包含: `name`, `start`, `end`, `length`, `diameter`, `slope`, `us_x`, `us_y`, `ds_x`, `ds_y`
        """)

    st.header("2. 控制面板")
    sim_hours = st.slider("模拟时长 (h)", 12, 48, 24)
    
    # 模拟触发按钮区
    if uploaded_file:
        st.divider()
        st.subheader("模拟执行")
        btn_hyd = st.button("▶️ 1. 运行水力模拟", type="primary", use_container_width=True)
        btn_wq = st.button("▶️ 2. 运行水质模拟", use_container_width=True, disabled=('hyd_res' not in st.session_state))

# --- 主逻辑 ---
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    df_pipe = process_uploaded_data(df_raw)
    
    if df_pipe is not None:
        # 构建图
        G = nx.DiGraph()
        for _, row in df_pipe.iterrows():
            G.add_edge(row['UpstreamNode'], row['DownstreamNode'], pipe_id=row['PipeID'])
        
        # 破环处理
        while not nx.is_directed_acyclic_graph(G):
            try:
                cycle = nx.find_cycle(G)
                G.remove_edge(*cycle[0])
            except: break
        topo_nodes = list(nx.topological_sort(G))

        # === 逻辑处理：水力模拟 ===
        if btn_hyd:
            with st.spinner("正在进行水力计算..."):
                all_nodes = list(G.nodes())
                node_inflows = generate_heterogeneous_inflows(all_nodes, hours=sim_hours)
                solver = VectorizedHydraulics()
                
                num_pipes = len(df_pipe)
                res_Q = np.zeros((num_pipes, sim_hours))
                res_v = np.zeros((num_pipes, sim_hours))
                res_h = np.zeros((num_pipes, sim_hours))
                
                for t in range(sim_hours):
                    node_acc = {n: node_inflows[n][t] for n in all_nodes}
                    pipe_flow_snap = {}
                    for u in topo_nodes:
                        total_in = node_acc[u]
                        out_edges = list(G.out_edges(u, data=True))
                        if not out_edges: continue
                        flow_per = total_in / len(out_edges)
                        for _, v_node, data in out_edges:
                            pid = data['pipe_id']
                            pipe_flow_snap[pid] = flow_per
                            if v_node in node_acc: node_acc[v_node] += flow_per
                    
                    curr_Q = np.array([pipe_flow_snap.get(pid, 0.0) for pid in df_pipe['PipeID']])
                    h, v, A, P = solver.solve_normal_depth(
                        curr_Q, df_pipe['Diameter'].values, df_pipe['Slope'].values, df_pipe['Manning'].values
                    )
                    res_Q[:, t] = curr_Q; res_v[:, t] = v; res_h[:, t] = h
                
                st.session_state['hyd_res'] = {'Q': res_Q, 'v': res_v, 'h': res_h, 'df': df_pipe}
                st.success("水力模拟完成！")
                st.rerun()

        # === 逻辑处理：水质模拟 ===
        if btn_wq and 'hyd_res' in st.session_state:
            with st.spinner("正在进行水质计算..."):
                hyd_res = st.session_state['hyd_res']
                df_p = hyd_res['df']
                sim_steps = hyd_res['Q'].shape[1]
                
                nodes_uniq = sorted(list(set(df_p['UpstreamNode']).union(set(df_p['DownstreamNode']))))
                n_map = {n: i for i, n in enumerate(nodes_uniq)}
                edge_src = [n_map[u] for u in df_p['UpstreamNode']]
                edge_dst = [n_map[v] for v in df_p['DownstreamNode']]
                edge_idx = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
                
                hyd_data = {
                    'Q': torch.tensor(hyd_res['Q'].T, dtype=torch.float32, device=device),
                    'v': torch.tensor(hyd_res['v'].T, dtype=torch.float32, device=device),
                    'h': torch.tensor(hyd_res['h'].T, dtype=torch.float32, device=device),
                    'L': torch.tensor(df_p['Length'].values, dtype=torch.float32, device=device).unsqueeze(0).expand(sim_steps, -1)
                }
                
                num_nodes = len(nodes_uniq)
                C_nodes = torch.zeros((num_nodes, 11), device=device) + 1e-6
                C_nodes[:, 3] = 6.0 
                asm = ASMKinetics().to(device)
                
                # 存储管段结果 (Time, Pipe, Params)
                history_pipes = [] 
                
                in_degs = [G.in_degree(n) for n in nodes_uniq]
                src_idxs = torch.tensor([i for i, d in enumerate(in_degs) if d == 0], dtype=torch.long, device=device)
                
                for t in range(sim_steps):
                    if len(src_idxs) > 0:
                        pattern = 1.0 + 0.5 * np.sin(2*np.pi*(t-8)/24)
                        C_nodes[src_idxs, 0] = 30.0 * pattern 
                        C_nodes[src_idxs, 1] = 150.0 * pattern 
                        C_nodes[src_idxs, 7] = 40.0 
                    
                    curr_v = hyd_data['v'][t]; curr_L = hyd_data['L'][t]; curr_Q = hyd_data['Q'][t]
                    res_time = torch.clamp((curr_L / (curr_v + 1e-4)) / 3600.0, max=1.0)
                    
                    C_in = C_nodes[edge_idx[0]]
                    hyd_state_t = {k: v[t].unsqueeze(1) for k, v in hyd_data.items() if k in ['v','h']}
                    
                    rates = asm.compute_rates(C_in, hyd_state_t)
                    C_out = C_in + rates * res_time.unsqueeze(1)
                    C_out = torch.clamp(C_out, min=1e-6)
                    
                    # 保存管段当前时刻的输出浓度
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
                
                st.session_state['wq_pipe_res'] = torch.stack(history_pipes, dim=0).numpy()
                st.success("水质模拟完成！")
                st.rerun()

        # === 界面展示 ===
        col_map, col_detail = st.columns([3, 2])
        
        with col_map:
            st.subheader("🗺️ 管网地图")
            if 'US_X' in df_pipe.columns:
                fig = create_interactive_map(df_pipe)
                
                # 关键：捕获选择事件
                selection = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
                
                # 解析选择 (修复版)
                selected_pipe_idx = None
                if selection and selection['selection']['points']:
                    # 获取被点击点的列表
                    points = selection['selection']['points']
                    # 遍历所有被点击的点
                    for point in points:
                        # 检查是否有 customdata，这是最可靠的标识
                        if 'customdata' in point:
                            selected_pipe_idx = point['customdata']
                            break
                        # 备用方案：如果只有 Pipe Select 层是 markers 模式，且有 pointIndex
                        # 我们在绘图时只有 Pipe Select 层设置了 customdata，所以上面的判断通常足够
                        # 如果没有 customdata，尝试直接读取 pointIndex，但需要确保不是点击了其他层
                        # 由于其他层 hoverinfo='skip' 且 mode='lines' 或 'markers' (Nodes)，
                        # 通常只有 Pipe Select 层会触发有效的 point selection
            else:
                st.warning("缺少坐标数据，无法绘图")

        with col_detail:
            st.subheader("📊 详细数据面板")
            
            if selected_pipe_idx is not None:
                # 确保索引是整数
                try:
                    selected_pipe_idx = int(selected_pipe_idx)
                    pipe_info = df_pipe.iloc[selected_pipe_idx]
                    pipe_id = pipe_info['PipeID']
                    st.info(f"当前选中管段: **{pipe_id}**")
                    
                    # 1. 基础属性
                    with st.expander("管段属性", expanded=False):
                        st.json({
                            "Length": f"{pipe_info['Length']} m",
                            "Diameter": f"{pipe_info['Diameter']} m",
                            "Slope": pipe_info['Slope'],
                            "Upstream": pipe_info['UpstreamNode'],
                            "Downstream": pipe_info['DownstreamNode']
                        })

                    # 2. 水力结果图表
                    if 'hyd_res' in st.session_state:
                        hyd = st.session_state['hyd_res']
                        ts = range(hyd['Q'].shape[1])
                        
                        fig_h, ax_h = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
                        
                        # 流量与流速
                        ax_h[0].plot(ts, hyd['Q'][selected_pipe_idx], 'b-', label='流量 Q')
                        ax_h[0].set_ylabel("Q (m³/s)")
                        ax_h[0].set_title("水力模拟结果")
                        ax_h[0].grid(True, alpha=0.3)
                        
                        ax2 = ax_h[0].twinx()
                        ax2.plot(ts, hyd['v'][selected_pipe_idx], 'orange', linestyle='--', label='流速 v')
                        ax2.set_ylabel("v (m/s)")
                        
                        # 水深
                        ax_h[1].plot(ts, hyd['h'][selected_pipe_idx], 'g-', label='水深 h')
                        ax_h[1].axhline(pipe_info['Diameter'], color='r', linestyle=':', label='管顶')
                        ax_h[1].set_ylabel("h (m)")
                        ax_h[1].set_xlabel("时间 (h)")
                        ax_h[1].legend()
                        ax_h[1].grid(True, alpha=0.3)
                        
                        st.pyplot(fig_h)
                    else:
                        st.info("暂无水力数据，请点击左侧运行模拟。")

                    # 3. 水质结果图表
                    if 'wq_pipe_res' in st.session_state:
                        wq = st.session_state['wq_pipe_res'] # Shape: (Time, Pipes, 11)
                        
                        fig_w, ax_w = plt.subplots(figsize=(6, 3))
                        # 绘制 DO (idx 3) 和 H2S (idx 6)
                        ax_w.plot(ts, wq[:, selected_pipe_idx, 3], 'b-', label='DO (氧)')
                        ax_w.plot(ts, wq[:, selected_pipe_idx, 6], 'r--', label='H2S (硫化物)')
                        ax_w.set_title("水质模拟结果")
                        ax_w.set_ylabel("浓度 (mg/L)")
                        ax_w.set_xlabel("时间 (h)")
                        ax_w.legend()
                        ax_w.grid(True, alpha=0.3)
                        
                        st.pyplot(fig_w)
                    elif 'hyd_res' in st.session_state:
                        st.info("暂无水质数据，请点击左侧运行模拟。")
                except Exception as e:
                    st.error(f"解析选中数据时出错: {e}")
            
            else:
                st.markdown("""
                <div style="text-align: center; padding: 50px; color: #666;">
                    👈 请点击地图上的<span style="color: red;">红色锚点</span><br>查看该管段的模拟结果
                </div>
                """, unsafe_allow_html=True)

else:
    st.info("👈 请在左侧上传 CSV 文件以开始。")
