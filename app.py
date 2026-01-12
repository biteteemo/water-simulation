import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import warnings
import io

# ==========================================
# 0. 配置与初始化
# ==========================================
st.set_page_config(page_title="城市管网水力与水质模拟", layout="wide")
st.markdown("""
<style>
.main {
    background-color: #f5f5f5;
}
</style>
""", unsafe_allow_html=True)

# 设置绘图参数
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] # 兼容中英文
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 设备选择
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. 核心类定义 (保持原有逻辑)
# ==========================================

class VectorizedHydraulics:
    def solve_normal_depth(self, Q_target, D, S, n):
        # 预处理：防止 Slope 为 0 或负数
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
        # 解包组分: XHw, Xs1, Xs2, SO, SF, Sac, SHS, SSO4, CH4, Sprop, H2
        XHw, Xs1, Xs2, SO, SF, Sac, SHS, SSO4, CH4, Sprop, H2 = [C[:, i:i+1] for i in range(11)]

        vel = hydraulic_state['v']; depth = hydraulic_state['h']
        
        depth_safe = torch.clamp(depth, min=1e-3)
        vel_safe = torch.clamp(vel, min=1e-3)
        
        # O'Connor-Dobbins 简化
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
# 2. 辅助函数
# ==========================================

def generate_sample_data():
    data = {
        'PipeID': [f'Pipe{i}' for i in range(10)],
        'UpstreamNode': ['Src1', 'Src2', 'N1', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7'],
        'DownstreamNode': ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N8', 'Outfall'],
        'Slope': [0.01, 0.015, 0.01, 0.008, 0.01, 0.012, 0.005, 0.005, 0.005, 0.002],
        'Diameter': [0.5, 0.4, 0.6, 0.5, 0.8, 0.8, 1.0, 1.0, 1.0, 1.2],
        'Length': [100, 120, 150, 100, 200, 180, 250, 150, 100, 300],
        'Manning': [0.013] * 10
    }
    return pd.DataFrame(data)

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
# 3. Streamlit 界面逻辑
# ==========================================

st.title("🏙️ 城市排水管网模拟系统 (Hydraulics + ASM)")
st.markdown("基于 Streamlit 的交互式模拟平台，集成水力计算与 ASM 水质动力学模型。")

# --- 侧边栏：数据上传 ---
with st.sidebar:
    st.header("1. 数据输入")
    data_source = st.radio("选择数据来源", ["使用测试数据", "上传 CSV 文件"])
    
    df_pipe = None
    if data_source == "上传 CSV 文件":
        uploaded_file = st.file_uploader("上传管网 CSV (包含 PipeID, UpstreamNode, DownstreamNode, Length, Diameter, Slope)", type=["csv"])
        if uploaded_file:
            df_pipe = pd.read_csv(uploaded_file)
    else:
        if st.button("加载测试数据"):
            df_pipe = generate_sample_data()
            st.success("测试数据已加载")

    st.header("2. 模拟设置")
    sim_hours = st.slider("模拟时长 (小时)", 12, 48, 24)
    
    st.info(f"计算设备: {device.upper()}")

# --- 主界面 ---
if df_pipe is not None:
    # 预处理数据
    if 'Manning' not in df_pipe.columns: df_pipe['Manning'] = 0.013
    df_pipe['UpstreamNode'] = df_pipe['UpstreamNode'].astype(str)
    df_pipe['DownstreamNode'] = df_pipe['DownstreamNode'].astype(str)

    # 构建图
    G = nx.DiGraph()
    for _, row in df_pipe.iterrows():
        G.add_edge(row['UpstreamNode'], row['DownstreamNode'], pipe_id=row['PipeID'])
    
    # 环路检测与处理
    if not nx.is_directed_acyclic_graph(G):
        st.warning("检测到环路，正在自动断开...")
        while not nx.is_directed_acyclic_graph(G):
            try:
                cycle = nx.find_cycle(G)
                G.remove_edge(*cycle[0])
            except: break
    
    # 拓扑排序
    try:
        topo_nodes = list(nx.topological_sort(G))
    except Exception as e:
        st.error(f"拓扑排序失败，请检查网络连通性: {e}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🌊 水力模拟", "🧪 水质模拟 (ASM)", "📊 数据概览"])

    # === Tab 1: 水力模拟 ===
    with tab1:
        st.subheader("全网水力计算")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            run_hyd = st.button("开始水力计算", type="primary")
        
        if run_hyd:
            with st.spinner("正在进行水力演算..."):
                # 初始化
                all_nodes = list(G.nodes())
                node_inflows = generate_heterogeneous_inflows(all_nodes, hours=sim_hours)
                solver = VectorizedHydraulics()
                
                num_pipes = len(df_pipe)
                # 结果存储
                res_Q = np.zeros((num_pipes, sim_hours))
                res_v = np.zeros((num_pipes, sim_hours))
                res_h = np.zeros((num_pipes, sim_hours))
                
                pipe_id_map = {pid: i for i, pid in enumerate(df_pipe['PipeID'])}
                
                progress_bar = st.progress(0)
                
                for t in range(sim_hours):
                    node_accumulation = {n: node_inflows[n][t] for n in all_nodes}
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
                    
                    # 矢量化计算
                    current_Q = np.array([pipe_flow_snapshot.get(pid, 0.0) for pid in df_pipe['PipeID']])
                    h, v, A, P = solver.solve_normal_depth(
                        current_Q, 
                        df_pipe['Diameter'].values, 
                        df_pipe['Slope'].values, 
                        df_pipe['Manning'].values
                    )
                    
                    res_Q[:, t] = current_Q
                    res_v[:, t] = v
                    res_h[:, t] = h
                    
                    progress_bar.progress((t + 1) / sim_hours)
                
                # 保存到 Session State
                st.session_state['hyd_results'] = {
                    'Q': res_Q, 'v': res_v, 'h': res_h, 
                    'df': df_pipe, 'G': G, 'topo_nodes': topo_nodes
                }
                st.success("水力计算完成！")

        # 结果展示
        if 'hyd_results' in st.session_state:
            res = st.session_state['hyd_results']
            
            st.markdown("### 关键路径分析")
            # 寻找最长路径用于绘图
            try:
                longest_path = nx.dag_longest_path(G)
                path_edges = []
                for i in range(len(longest_path)-1):
                    u, v = longest_path[i], longest_path[i+1]
                    path_edges.append(G.get_edge_data(u, v)['pipe_id'])
                
                selected_pipe = st.selectbox("选择要查看的管段 (默认显示主干路径末端)", path_edges, index=len(path_edges)-1)
                idx = df_pipe[df_pipe['PipeID'] == selected_pipe].index[0]
                
                fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                times = range(sim_hours)
                
                ax[0].plot(times, res['Q'][idx], 'o-', color='#1f77b4')
                ax[0].set_ylabel("流量 Q (m³/s)")
                ax[0].set_title(f"管段 {selected_pipe} 水力要素变化")
                
                ax[1].plot(times, res['v'][idx], 's--', color='#ff7f0e')
                ax[1].set_ylabel("流速 v (m/s)")
                
                ax[2].plot(times, res['h'][idx], '^:', color='#2ca02c')
                ax[2].axhline(y=df_pipe.iloc[idx]['Diameter'], color='r', linestyle='--', alpha=0.3, label='管径')
                ax[2].set_ylabel("水深 h (m)")
                ax[2].set_xlabel("时间 (h)")
                ax[2].legend()
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"绘图错误: {e}")

    # === Tab 2: 水质模拟 ===
    with tab2:
        st.subheader("ASM 动态水质模拟")
        
        if 'hyd_results' not in st.session_state:
            st.warning("请先在 Tab 1 完成水力计算。")
        else:
            st.markdown("模型将基于上一步计算的动态流速和水深，模拟污染物在管网中的反应与迁移。")
            
            # 准备 PyTorch 数据
            hyd_res = st.session_state['hyd_results']
            df_p = hyd_res['df']
            
            # 节点映射
            all_nodes_unique = sorted(list(set(df_p['UpstreamNode']).union(set(df_p['DownstreamNode']))))
            node_map = {n: i for i, n in enumerate(all_nodes_unique)}
            num_nodes = len(all_nodes_unique)
            
            # 识别出流节点 (Outfall)
            out_degrees = [G.out_degree(n) for n in all_nodes_unique]
            outfall_nodes = [n for n, d in zip(all_nodes_unique, out_degrees) if d == 0]
            
            run_asm = st.button("开始水质模拟", type="primary")
            
            if run_asm:
                with st.spinner("正在进行生化反应模拟 (PyTorch)..."):
                    # 构建 Tensor 数据
                    edge_src = [node_map[u] for u in df_p['UpstreamNode']]
                    edge_dst = [node_map[v] for v in df_p['DownstreamNode']]
                    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
                    
                    # 水力张量 [T, N_pipes]
                    hyd_data = {
                        'Q': torch.tensor(hyd_res['Q'], dtype=torch.float32, device=device),
                        'v': torch.tensor(hyd_res['v'], dtype=torch.float32, device=device),
                        'h': torch.tensor(hyd_res['h'], dtype=torch.float32, device=device),
                        'D': torch.tensor(df_p['Diameter'].values, dtype=torch.float32, device=device).unsqueeze(0).repeat(sim_hours, 1),
                        'S': torch.tensor(df_p['Slope'].values, dtype=torch.float32, device=device).unsqueeze(0).repeat(sim_hours, 1),
                        'L': torch.tensor(df_p['Length'].values, dtype=torch.float32, device=device).unsqueeze(0).repeat(sim_hours, 1)
                    }
                    
                    # 边界条件 (简化：所有入流节点有固定污染模式)
                    in_degrees = [G.in_degree(n) for n in all_nodes_unique]
                    inflow_nodes_idx = [i for i, d in enumerate(in_degrees) if d == 0]
                    inflow_idxs_t = torch.tensor(inflow_nodes_idx, dtype=torch.long, device=device)
                    
                    # 生成边界浓度 [T, N_in, 11]
                    T = sim_hours
                    inflow_boundary = torch.zeros((T, len(inflow_nodes_idx), 11), device=device)
                    time_vec = torch.linspace(0, T, T, device=device)
                    pattern = 1.0 + 0.5 * torch.sin(2 * np.pi * (time_vec - 8) / 24)
                    
                    for i in range(len(inflow_nodes_idx)):
                        inflow_boundary[:, i, 0] = 30.0 * pattern # XHw
                        inflow_boundary[:, i, 1] = 150.0 * pattern # Xs1 (COD)
                        inflow_boundary[:, i, 3] = 2.0   # DO
                        inflow_boundary[:, i, 6] = 0.0   # H2S
                        inflow_boundary[:, i, 7] = 50.0  # Sulfate
                    
                    # === 动态求解器 (简化版嵌入) ===
                    C_nodes = torch.zeros((num_nodes, 11), device=device) + 1e-6
                    C_nodes[:, 3] = 5.0 # 初始 DO
                    history_nodes = []
                    asm = ASMKinetics().to(device)
                    
                    asm_progress = st.progress(0)
                    
                    for t in range(T):
                        # 1. 边界
                        if len(inflow_nodes_idx) > 0:
                            C_nodes[inflow_idxs_t] = inflow_boundary[t]
                        
                        # 2. 管道反应
                        curr_v = hyd_data['v'][t]; curr_L = hyd_data['L'][t]
                        curr_Q = hyd_data['Q'][t]
                        
                        # 停留时间
                        res_time = torch.clamp((curr_L / (curr_v + 1e-4)) / 3600.0, max=1.0)
                        
                        src_idx = edge_index[0]; dst_idx = edge_index[1]
                        C_pipe_in = C_nodes[src_idx]
                        
                        hyd_state_t = {k: v[t].unsqueeze(1) for k, v in hyd_data.items() if k in ['v','h','D','S']}
                        rates = asm.compute_rates(C_pipe_in, hyd_state_t)
                        C_pipe_out = C_pipe_in + rates * res_time.unsqueeze(1)
                        C_pipe_out = torch.clamp(C_pipe_out, min=1e-6)
                        
                        # 3. 节点混合
                        mass_flux = C_pipe_out * curr_Q.unsqueeze(1)
                        total_mass = torch.zeros((num_nodes, 11), device=device)
                        total_Q = torch.zeros((num_nodes, 1), device=device)
                        
                        total_mass.index_add_(0, dst_idx, mass_flux)
                        total_Q.index_add_(0, dst_idx, curr_Q.unsqueeze(1))
                        
                        mask_flow = (total_Q > 1e-6).squeeze()
                        unique_dst = torch.unique(dst_idx)
                        # 仅更新下游节点
                        valid_dst = unique_dst[mask_flow[unique_dst]]
                        C_nodes[valid_dst] = total_mass[valid_dst] / total_Q[valid_dst]
                        
                        history_nodes.append(C_nodes.clone().cpu())
                        asm_progress.progress((t+1)/T)
                    
                    sim_results = torch.stack(history_nodes, dim=0).numpy() # [T, N_nodes, 11]
                    st.session_state['wq_results'] = sim_results
                    st.session_state['outfall_nodes'] = outfall_nodes
                    st.session_state['node_map'] = node_map
                    st.success("水质模拟完成！")

            # 水质结果展示
            if 'wq_results' in st.session_state:
                wq_res = st.session_state['wq_results']
                outfalls = st.session_state['outfall_nodes']
                n_map = st.session_state['node_map']
                
                target_node = st.selectbox("选择观测节点 (推荐 Outfall)", outfalls + list(n_map.keys()))
                n_idx = n_map[target_node]
                
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                times = range(sim_hours)
                
                # Plot H2S (idx 6), DO (idx 3), COD_fast (idx 1)
                ax2.plot(times, wq_res[:, n_idx, 6], 'r-o', label='硫化物 (H2S)', linewidth=2)
                ax2.plot(times, wq_res[:, n_idx, 3], 'b--s', label='溶解氧 (DO)')
                ax2.plot(times, wq_res[:, n_idx, 1] / 10.0, 'g:', label='快速COD / 10')
                
                ax2.set_title(f"节点 {target_node} 水质变化趋势")
                ax2.set_xlabel("时间 (h)")
                ax2.set_ylabel("浓度 (mg/L)")
                ax2.legend()
                st.pyplot(fig2)
                
                st.info("说明: 红色曲线代表硫化物产生（恶臭来源），蓝色代表溶解氧消耗。")

    # === Tab 3: 数据概览 ===
    with tab3:
        st.subheader("管网基础数据")
        st.dataframe(df_pipe)
        
        if 'hyd_results' in st.session_state:
            st.subheader("T=12h (中午) 水力快照")
            t_idx = 12 if sim_hours >= 12 else 0
            df_snap = df_pipe.copy()
            df_snap['Q_12h'] = st.session_state['hyd_results']['Q'][:, t_idx]
            df_snap['v_12h'] = st.session_state['hyd_results']['v'][:, t_idx]
            df_snap['h_12h'] = st.session_state['hyd_results']['h'][:, t_idx]
            st.dataframe(df_snap)

else:
    st.info("👈 请在左侧上传数据或加载测试数据以开始。")
