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
st.set_page_config(page_title="城市管网模拟 (Custom Data)", layout="wide")
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
h1 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. 核心计算类 (保持不变)
# ==========================================

class VectorizedHydraulics:
    def solve_normal_depth(self, Q_target, D, S, n):
        # 强制 S > 0 防止除零
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
# 2. 数据处理函数 (针对你的CSV格式)
# ==========================================

def process_uploaded_data(df):
    """
    处理用户上传的特定格式 CSV
    列映射: name->PipeID, start->UpstreamNode, end->DownstreamNode, etc.
    """
    # 1. 列名映射字典
    col_map = {
        'name': 'PipeID',
        'start': 'UpstreamNode',
        'end': 'DownstreamNode',
        'length': 'Length',
        'diameter': 'Diameter',
        'slope': 'Slope',
        'us_x': 'US_X', 'us_y': 'US_Y',
        'ds_x': 'DS_X', 'ds_y': 'DS_Y'
    }
    
    # 2. 重命名
    df = df.rename(columns=col_map)
    
    # 3. 检查必要列
    required_cols = ['PipeID', 'UpstreamNode', 'DownstreamNode', 'Length', 'Diameter', 'Slope']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"数据缺少必要列 (映射后): {missing}")
        return None

    # 4. 数据清洗
    # 强制节点ID为字符串
    df['UpstreamNode'] = df['UpstreamNode'].astype(str)
    df['DownstreamNode'] = df['DownstreamNode'].astype(str)
    
    # 修正 Slope = 0 的情况 (防止曼宁公式报错)
    zero_slope_count = (df['Slope'] <= 0).sum()
    if zero_slope_count > 0:
        st.warning(f"检测到 {zero_slope_count} 条管段坡度为 0，已自动修正为 0.001 以进行计算。")
        df['Slope'] = df['Slope'].clip(lower=0.001)
        
    # 默认曼宁系数
    if 'Manning' not in df.columns:
        df['Manning'] = 0.013
        
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
# 3. Streamlit 界面
# ==========================================

st.title("🏙️ 城市排水管网模拟系统")
st.markdown("支持自定义 CSV 格式 (包含坐标数据)")

# --- 侧边栏 ---
with st.sidebar:
    st.header("1. 数据导入")
    uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
    
    # 提供示例数据格式说明
    with st.expander("查看支持的数据格式"):
        st.markdown("""
        您的 CSV 应包含以下列 (大小写不敏感):
        - `name`: 管段ID
        - `start`: 上游节点ID
        - `end`: 下游节点ID
        - `length`: 长度 (m)
        - `diameter`: 管径 (m)
        - `slope`: 坡度
        - `us_x`, `us_y`: 上游坐标 (可选，用于绘图)
        - `ds_x`, `ds_y`: 下游坐标 (可选，用于绘图)
        """)

    st.header("2. 模拟参数")
    sim_hours = st.slider("模拟时长 (h)", 12, 48, 24)
    st.info(f"计算设备: {device}")

# --- 主逻辑 ---
if uploaded_file:
    # 读取并处理数据
    df_raw = pd.read_csv(uploaded_file)
    df_pipe = process_uploaded_data(df_raw)
    
    if df_pipe is not None:
        # 构建图
        G = nx.DiGraph()
        for _, row in df_pipe.iterrows():
            # 添加边属性，包括坐标以便绘图
            edge_attrs = {'pipe_id': row['PipeID']}
            if 'US_X' in row and 'DS_X' in row:
                edge_attrs['pos_src'] = (row['US_X'], row['US_Y'])
                edge_attrs['pos_dst'] = (row['DS_X'], row['DS_Y'])
            G.add_edge(row['UpstreamNode'], row['DownstreamNode'], **edge_attrs)
        
        # 环路处理
        if not nx.is_directed_acyclic_graph(G):
            st.warning("检测到环路，正在自动断开以支持水力计算...")
            while not nx.is_directed_acyclic_graph(G):
                try:
                    cycle = nx.find_cycle(G)
                    G.remove_edge(*cycle[0])
                except: break
        
        topo_nodes = list(nx.topological_sort(G))
        
        # 界面标签页
        tab_map, tab_hyd, tab_asm, tab_data = st.tabs(["🗺️ 管网地图", "🌊 水力模拟", "🧪 水质模拟", "📄 数据概览"])

        # === Tab 1: 管网地图 ===
        with tab_map:
            st.subheader("管网平面分布图")
            if 'US_X' in df_pipe.columns:
                fig_map, ax_map = plt.subplots(figsize=(10, 8))
                
                # 绘制管段
                for _, row in df_pipe.iterrows():
                    ax_map.plot([row['US_X'], row['DS_X']], [row['US_Y'], row['DS_Y']], 
                                color='gray', alpha=0.5, linewidth=1)
                
                # 绘制节点 (简单的散点)
                ax_map.scatter(df_pipe['US_X'], df_pipe['US_Y'], s=10, c='blue', alpha=0.6, label='Nodes')
                
                ax_map.set_title(f"管网拓扑结构 (共 {len(df_pipe)} 条管段)")
                ax_map.set_xlabel("X Coordinate")
                ax_map.set_ylabel("Y Coordinate")
                ax_map.axis('equal')
                st.pyplot(fig_map)
            else:
                st.warning("数据中未检测到 us_x, us_y 等坐标列，无法绘制地图。")

        # === Tab 2: 水力模拟 ===
        with tab_hyd:
            col1, col2 = st.columns([1, 4])
            if col1.button("▶️ 开始水力计算", type="primary"):
                with st.spinner("正在求解曼宁方程..."):
                    all_nodes = list(G.nodes())
                    node_inflows = generate_heterogeneous_inflows(all_nodes, hours=sim_hours)
                    solver = VectorizedHydraulics()
                    
                    num_pipes = len(df_pipe)
                    res_Q = np.zeros((num_pipes, sim_hours))
                    res_v = np.zeros((num_pipes, sim_hours))
                    res_h = np.zeros((num_pipes, sim_hours))
                    
                    prog_bar = st.progress(0)
                    
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
                                if v_node in node_acc:
                                    node_acc[v_node] += flow_per
                        
                        curr_Q = np.array([pipe_flow_snap.get(pid, 0.0) for pid in df_pipe['PipeID']])
                        h, v, A, P = solver.solve_normal_depth(
                            curr_Q, df_pipe['Diameter'].values, df_pipe['Slope'].values, df_pipe['Manning'].values
                        )
                        
                        res_Q[:, t] = curr_Q
                        res_v[:, t] = v
                        res_h[:, t] = h
                        prog_bar.progress((t+1)/sim_hours)
                    
                    st.session_state['hyd_res'] = {'Q': res_Q, 'v': res_v, 'h': res_h, 'df': df_pipe}
                    st.success("计算完成！")
            
            if 'hyd_res' in st.session_state:
                res = st.session_state['hyd_res']
                st.markdown("#### 结果分析")
                
                # 选择管段
                sel_pipe = st.selectbox("选择管段查看详情", df_pipe['PipeID'].values)
                idx = df_pipe[df_pipe['PipeID'] == sel_pipe].index[0]
                
                fig_h, ax_h = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
                ts = range(sim_hours)
                
                ax_h[0].plot(ts, res['Q'][idx], color='#1f77b4', marker='.')
                ax_h[0].set_ylabel("流量 Q (m3/s)")
                ax_h[0].set_title(f"管段 {sel_pipe} 水力过程线")
                
                ax_h[1].plot(ts, res['v'][idx], color='#ff7f0e', marker='.')
                ax_h[1].set_ylabel("流速 v (m/s)")
                
                ax_h[2].plot(ts, res['h'][idx], color='#2ca02c', marker='.')
                ax_h[2].axhline(df_pipe.iloc[idx]['Diameter'], ls='--', c='r', alpha=0.5, label='管顶')
                ax_h[2].set_ylabel("水深 h (m)")
                ax_h[2].legend()
                
                st.pyplot(fig_h)

        # === Tab 3: 水质模拟 ===
        with tab_asm:
            if 'hyd_res' not in st.session_state:
                st.warning("请先完成水力模拟。")
            else:
                if st.button("▶️ 开始 ASM 水质模拟", type="primary"):
                    hyd_res = st.session_state['hyd_res']
                    df_p = hyd_res['df']
                    
                    # 准备拓扑
                    nodes_uniq = sorted(list(set(df_p['UpstreamNode']).union(set(df_p['DownstreamNode']))))
                    n_map = {n: i for i, n in enumerate(nodes_uniq)}
                    
                    edge_src = [n_map[u] for u in df_p['UpstreamNode']]
                    edge_dst = [n_map[v] for v in df_p['DownstreamNode']]
                    edge_idx = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
                    
                    # ==========================================
                    # 修复点：数据形状对齐
                    # 目标形状均为: (sim_hours, num_pipes)
                    # ==========================================
                    sim_steps = hyd_res['Q'].shape[1] # 获取实际模拟时长列数
                    num_pipes = len(df_p)

                    hyd_data = {
                        # 注意这里添加了 .T 进行转置，将 (Pipe, Time) 变为 (Time, Pipe)
                        'Q': torch.tensor(hyd_res['Q'].T, dtype=torch.float32, device=device),
                        'v': torch.tensor(hyd_res['v'].T, dtype=torch.float32, device=device),
                        'h': torch.tensor(hyd_res['h'].T, dtype=torch.float32, device=device),
                        
                        # 静态属性扩展为 (Time, Pipe)
                        'D': torch.tensor(df_p['Diameter'].values, dtype=torch.float32, device=device).unsqueeze(0).expand(sim_steps, -1),
                        'S': torch.tensor(df_p['Slope'].values, dtype=torch.float32, device=device).unsqueeze(0).expand(sim_steps, -1),
                        'L': torch.tensor(df_p['Length'].values, dtype=torch.float32, device=device).unsqueeze(0).expand(sim_steps, -1)
                    }
                    
                    # 模拟循环
                    num_nodes = len(nodes_uniq)
                    C_nodes = torch.zeros((num_nodes, 11), device=device) + 1e-6
                    C_nodes[:, 3] = 6.0 # 初始 DO
                    
                    asm = ASMKinetics().to(device)
                    history = []
                    prog_asm = st.progress(0)
                    
                    # 简单的源头入流
                    in_degs = [G.in_degree(n) for n in nodes_uniq]
                    src_idxs = torch.tensor([i for i, d in enumerate(in_degs) if d == 0], dtype=torch.long, device=device)
                    
                    for t in range(sim_steps):
                        # 边界条件
                        if len(src_idxs) > 0:
                            pattern = 1.0 + 0.5 * np.sin(2*np.pi*(t-8)/24)
                            C_nodes[src_idxs, 0] = 30.0 * pattern # XHw
                            C_nodes[src_idxs, 1] = 150.0 * pattern # COD
                            C_nodes[src_idxs, 7] = 40.0 # Sulfate
                        
                        # 反应
                        # 现在 hyd_data['v'][t] 取出的是 (num_pipes,) 形状的向量，与 L 形状一致
                        curr_v = hyd_data['v'][t]
                        curr_L = hyd_data['L'][t]
                        curr_Q = hyd_data['Q'][t]
                        
                        # 这里的除法现在是安全的：(num_pipes,) / (num_pipes,)
                        res_time = torch.clamp((curr_L / (curr_v + 1e-4)) / 3600.0, max=1.0)
                        
                        C_in = C_nodes[edge_idx[0]] # (num_pipes, 11)
                        
                        # 准备水力状态供 kinetics 使用，需要 unsqueeze 变成 (num_pipes, 1)
                        hyd_state_t = {k: v[t].unsqueeze(1) for k, v in hyd_data.items() if k in ['v','h','D','S']}
                        
                        rates = asm.compute_rates(C_in, hyd_state_t)
                        C_out = C_in + rates * res_time.unsqueeze(1)
                        C_out = torch.clamp(C_out, min=1e-6)
                        
                        # 混合
                        mass = C_out * curr_Q.unsqueeze(1)
                        tot_m = torch.zeros((num_nodes, 11), device=device)
                        tot_q = torch.zeros((num_nodes, 1), device=device)
                        tot_m.index_add_(0, edge_idx[1], mass)
                        tot_q.index_add_(0, edge_idx[1], curr_Q.unsqueeze(1))
                        
                        mask = (tot_q > 1e-6).squeeze()
                        valid_dst = torch.unique(edge_idx[1])
                        # 过滤掉无效节点
                        valid_dst = valid_dst[mask[valid_dst]]
                        
                        if len(valid_dst) > 0:
                            C_nodes[valid_dst] = tot_m[valid_dst] / tot_q[valid_dst]
                        
                        history.append(C_nodes.clone().cpu())
                        prog_asm.progress((t+1)/sim_steps)
                        
                    st.session_state['wq_res'] = torch.stack(history, dim=0).numpy()
                    st.session_state['nodes_list'] = nodes_uniq
                    st.success("水质模拟完成")

                if 'wq_res' in st.session_state:
                    wq = st.session_state['wq_res']
                    nodes = st.session_state['nodes_list']
                    
                    # 找出 Outfall (出度为0)
                    outfalls = [n for n in nodes if G.out_degree(n) == 0]
                    # 如果没有明确的出水口，显示所有节点
                    display_nodes = outfalls if outfalls else nodes
                    
                    sel_node = st.selectbox("选择观测节点", display_nodes)
                    n_idx = nodes.index(sel_node)
                    
                    fig_w, ax_w = plt.subplots(figsize=(10, 5))
                    ax_w.plot(wq[:, n_idx, 6], 'r-', label='H2S (Sulfide)')
                    ax_w.plot(wq[:, n_idx, 3], 'b--', label='DO (Oxygen)')
                    ax_w.set_ylabel("浓度 (mg/L)")
                    ax_w.set_xlabel("时间 (h)")
                    ax_w.set_title(f"节点 {sel_node} 水质")
                    ax_w.legend()
                    st.pyplot(fig_w)

        # === Tab 4: 数据概览 ===
        with tab_data:
            st.dataframe(df_pipe)

else:
    st.info("👈 请在左侧上传 CSV 文件以开始。")

