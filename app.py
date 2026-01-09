import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time

# 设置页面布局为宽屏模式
st.set_page_config(layout="wide", page_title="供热管网数字孪生系统")

# --- 1. 数据生成与初始化 ---

@st.cache_data
def generate_mock_data():
    """生成模拟的管网拓扑数据和历史运行数据"""
    # 生成管段拓扑 (简单的星型+环形结构)
    # 坐标范围模拟某城市区域
    base_lat, base_lon = 39.90, 116.40
    
    pipes_data = []
    num_pipes = 20
    
    for i in range(num_pipes):
        # 随机生成起点和终点
        start_lat = base_lat + np.random.uniform(-0.05, 0.05)
        start_lon = base_lon + np.random.uniform(-0.05, 0.05)
        end_lat = start_lat + np.random.uniform(-0.01, 0.01)
        end_lon = start_lon + np.random.uniform(-0.01, 0.01)
        
        pipes_data.append({
            "PipeID": f"P-{1000+i}",
            "StartLat": start_lat, "StartLon": start_lon,
            "EndLat": end_lat, "EndLon": end_lon,
            "Diameter": np.random.choice([300, 500, 800]),  # 管径 mm
            "Length": np.random.randint(100, 1000),         # 长度 m
            "InstallYear": np.random.randint(1990, 2020)
        })
    
    df_pipe = pd.DataFrame(pipes_data)
    return df_pipe

def generate_timeseries_data(pipe_id):
    """为特定管段生成模拟的时序数据"""
    dates = pd.date_range(start="2023-12-01", periods=24, freq="H")
    # 模拟温度 (供水)
    temp_supply = 85 + np.random.randn(24) * 2
    # 模拟压力 (MPa)
    pressure = 0.8 + np.random.randn(24) * 0.05
    # 模拟流量 (t/h)
    flow = 120 + np.sin(np.arange(24)/3) * 20 + np.random.randn(24) * 5
    
    return pd.DataFrame({
        "Time": dates,
        "Temperature": temp_supply,
        "Pressure": pressure,
        "Flow": flow
    }).set_index("Time")

# 加载基础数据
df_pipe = generate_mock_data()

# --- 2. Session State 初始化 ---

# 初始化当前选中的管段ID
if 'selected_pipe_id' not in st.session_state:
    st.session_state['selected_pipe_id'] = df_pipe.iloc[0]['PipeID']

# 初始化下拉框的 Key
if 'pipe_selector' not in st.session_state:
    st.session_state['pipe_selector'] = df_pipe.iloc[0]['PipeID']

# 初始化模拟状态
if 'simulation_running' not in st.session_state:
    st.session_state['simulation_running'] = False

# --- 3. 侧边栏 ---
with st.sidebar:
    st.title("⚙️ 系统控制台")
    st.info("数字孪生供热管网演示系统")
    
    st.markdown("### 全局参数设置")
    ambient_temp = st.slider("环境温度 (°C)", -20, 10, -5)
    heat_source_temp = st.slider("热源出水温度 (°C)", 70, 110, 90)
    
    st.markdown("### 显示设置")
    show_labels = st.checkbox("显示管段标签", value=False)
    map_style = st.selectbox("地图风格", ["mapbox://styles/mapbox/dark-v10", "mapbox://styles/mapbox/light-v10"])

# --- 4. 主界面布局 ---

st.title("🔥 智慧供热管网数字孪生平台")

# 创建两列布局：左侧地图，右侧详情
col_map, col_details = st.columns([3, 2])

# ==============================================================================
# 左侧：地图区域 (核心修改区域)
# ==============================================================================
with col_map:
    st.subheader("🗺️ 管网地图")
    
    # 准备 PyDeck 数据
    # 为了让 PyDeck 能够绘制线条，需要将起终点坐标整理成特定格式
    df_map = df_pipe.copy()
    df_map['path'] = df_map.apply(lambda row: [[row['StartLon'], row['StartLat']], [row['EndLon'], row['EndLat']]], axis=1)
    
    # 根据当前选中状态给管段上色
    # 选中：红色 [255, 0, 0]，未选中：青色 [0, 128, 200]
    df_map['color'] = df_map['PipeID'].apply(
        lambda x: [255, 0, 0, 200] if x == st.session_state['selected_pipe_id'] else [0, 128, 200, 140]
    )
    
    # 根据管径设置线宽
    df_map['width'] = df_map['Diameter'] / 10

    # 定义图层
    layer = pdk.Layer(
        "PathLayer",
        df_map,
        pickable=True,
        get_path="path",
        get_width="width",
        get_color="color",
        width_scale=1,
        width_min_pixels=2,
        auto_highlight=True,
        highlight_color=[255, 255, 0, 255], # 鼠标悬停高亮黄色
    )

    # 初始视角
    view_state = pdk.ViewState(
        latitude=df_map['StartLat'].mean(),
        longitude=df_map['StartLon'].mean(),
        zoom=12,
        pitch=45,
    )

    # 渲染地图
    tooltip = {
        "html": "<b>管段ID:</b> {PipeID} <br/> <b>管径:</b> {Diameter} mm",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }

    # 使用 pydeck_chart 并开启选择功能
    map_chart = st.pydeck_chart(
        pdk.Deck(
            layers=[layer], 
            initial_view_state=view_state, 
            map_style=map_style,
            tooltip=tooltip
        ),
        on_select="rerun",  # 关键：选中后重新运行脚本
        selection_mode="single-object", # 单选模式
        use_container_width=True
    )

    # --- 地图点击交互逻辑 ---
    if map_chart.selection:
        indices = map_chart.selection.get("indices")
        if indices:
            # 获取点击的行索引
            clicked_idx = indices[0]
            # 获取对应的 PipeID
            clicked_id = df_map.iloc[clicked_idx]['PipeID']
            
            # 如果点击的 ID 与当前存储的 ID 不一致，则更新状态
            if clicked_id != st.session_state.get('pipe_selector'):
                st.session_state['pipe_selector'] = clicked_id
                st.session_state['selected_pipe_id'] = clicked_id
                st.rerun() # 强制刷新以更新右侧下拉框

# ==============================================================================
# 右侧：详情与图表区域 (核心修改区域)
# ==============================================================================
with col_details:
    st.subheader("📈 模拟与分析")
    
    # 模拟控制
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶ 开始水力模拟", use_container_width=True):
            st.session_state['simulation_running'] = True
            with st.spinner('正在解算水力矩阵...'):
                time.sleep(1.5) # 假装在计算
            st.success("模拟完成！")
    with col_btn2:
        if st.button("⏹ 重置状态", use_container_width=True):
            st.session_state['simulation_running'] = False
            st.rerun()

    st.divider()

    # --- 管道选择器逻辑 ---
    all_ids = df_pipe['PipeID'].values.tolist()
    
    # 定义回调函数：当用户手动改变下拉框时触发
    def on_selector_change():
        st.session_state['selected_pipe_id'] = st.session_state['pipe_selector']

    # 确保当前 pipe_selector 的值在列表中，防止报错
    try:
        current_index = all_ids.index(st.session_state['pipe_selector'])
    except (ValueError, KeyError):
        current_index = 0
        st.session_state['pipe_selector'] = all_ids[0]

    # 渲染下拉框
    # key="pipe_selector" 实现了与 Session State 的双向绑定
    selected_id = st.selectbox(
        "选择管段查看详情:", 
        options=all_ids,
        index=current_index, # 显式指定索引，确保视觉同步
        key="pipe_selector", 
        on_change=on_selector_change
    )
    
    # 再次确保 selected_pipe_id 同步 (双重保险)
    st.session_state['selected_pipe_id'] = selected_id

    # --- 显示选中管段的属性 ---
    pipe_info = df_pipe[df_pipe['PipeID'] == selected_id].iloc[0]
    
    st.markdown(f"#### 🏷️ 管段信息: {selected_id}")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("管径", f"{pipe_info['Diameter']} mm")
    c2.metric("长度", f"{pipe_info['Length']} m")
    c3.metric("敷设年份", f"{pipe_info['InstallYear']}")

    # --- 显示选中管段的图表 ---
    st.markdown("#### 📊 实时运行曲线")
    
    # 获取模拟时序数据
    df_chart = generate_timeseries_data(selected_id)
    
    # 使用 Tabs 切换不同图表
    tab1, tab2, tab3 = st.tabs(["温度趋势", "压力分布", "流量监控"])
    
    with tab1:
        st.line_chart(df_chart['Temperature'], color="#FF4B4B")
    with tab2:
        st.area_chart(df_chart['Pressure'], color="#1F77B4")
    with tab3:
        st.bar_chart(df_chart['Flow'], color="#2CA02C")

# --- 5. 底部状态栏 ---
st.markdown("---")
st.caption(f"系统状态: {'🟢 在线运行' if st.session_state['simulation_running'] else '⚪ 待机中'} | 数据最后更新: {time.strftime('%Y-%m-%d %H:%M:%S')}")
