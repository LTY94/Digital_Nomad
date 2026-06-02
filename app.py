import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 網頁基本設定 (必須放在程式碼第一行)
# ==========================================
st.set_page_config(
    page_title="全球數位遊牧城市大數據探勘系統",
    page_icon="🌍",
    layout="wide"
)

# ==========================================
# 2. 建立/模擬數位遊牧大數據 (欄位豐富、適合探勘)
# ==========================================
@st.cache_data
def load_data():
    # 這裡模擬真實從 Nomad List 爬取下來的結構化資料
    np.random.seed(42)
    cities = [
        "Changgu (Bali)", "Bangkok", "Chiang Mai", "Medellin", "Lisbon", 
        "Barcelona", "Taipei", "Tokyo", "Seoul", "Berlin", 
        "Buenos Aires", "Cape Town", "Budapest", "Mexico City", "Da Nang"
    ] * 5  # 擴展資料量
    
    data = {
        "City": [f"{c}_{i}" for i, c in enumerate(cities)],
        "Region": np.random.choice(["東南亞", "東北亞", "歐洲", "美洲", "非洲"], len(cities)),
        "Cost_of_Living_USD": np.random.randint(800, 4500, len(cities)),
        "Internet_Speed_Mbps": np.random.randint(20, 500, len(cities)),
        "Safety_Score": np.random.uniform(1.0, 5.0, len(cities)),
        "Entertainment_Score": np.random.uniform(1.0, 5.0, len(cities)),
        "Co_Working_Spaces": np.random.randint(2, 80, len(cities)),
        "English_Proficiency": np.random.uniform(1.0, 5.0, len(cities))
    }
    df = pd.DataFrame(data)
    # 清洗資料：確保台北和東京的數據不要太離譜
    df.loc[df['City'].str.contains("Taipei"), 'Internet_Speed_Mbps'] = np.random.randint(300, 600)
    df.loc[df['City'].str.contains("Taipei"), 'Safety_Score'] = np.random.uniform(4.5, 5.0)
    return df

df = load_data()

# ==========================================
# 3. 側邊欄選單設計 (Sidebar Navigation)
# ==========================================
st.sidebar.title("🌍 導覽選單")
st.sidebar.markdown("請選擇您要查看的資料探勘維度：")
page = st.sidebar.radio(
    "分析面向",
    ["📊 城市基本數據分布", "💰 成本與網速關聯分析", "🤖 機器學習：城市 AI 分群", "🔍 頂級遊牧城市篩選器"]
)

# 主網頁大標題
st.title("🌍 全球數位遊牧（Digital Nomad）城市生活品質與成本綜合評估系統")
st.markdown("---")

# ==========================================
# 頁面 1：📊 城市基本數據分布 (長條圖、圓餅圖)
# ==========================================
if page == "📊 城市基本數據分布":
    st.header("📌 各大區域城市數據統計")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💡 各區域遊牧城市佔比 (圓餅圖)")
        region_counts = df['Region'].value_counts().reset_index()
        fig_pie = px.pie(region_counts, values='count', names='Region', hole=0.4,
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.subheader("💰 各區域平均生活成本 (USD)")
        avg_cost = df.groupby('Region')['Cost_of_Living_USD'].mean().reset_index().sort_values(by='Cost_of_Living_USD')
        fig_bar = px.bar(avg_cost, x='Region', y='Cost_of_Living_USD', 
                         color='Cost_of_Living_USD', color_continuous_scale='Viridis',
                         labels={'Cost_of_Living_USD': '平均月花費 (USD)'})
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 原始探勘資料集摘要")
    st.dataframe(df.style.background_gradient(cmap='Blues', subset=['Cost_of_Living_USD', 'Internet_Speed_Mbps']))

# ==========================================
# 頁面 2：💰 成本與網速關聯分析 (散佈圖、趨勢線)
# ==========================================
elif page == "💰 成本與網速關聯分析":
    st.header("📌 尋找高 C/P 值天堂：網速 vs 生活成本")
    st.markdown("> **探勘目標：** 找出位於**右下角**（生活成本低、網路速度極快）的夢幻遊牧城市。")
    
    # 互動式篩選器
    selected_region = st.multiselect("篩選區域", options=df['Region'].unique(), default=df['Region'].unique())
    filtered_df = df[df['Region'].isin(selected_region)]
    
    # 繪製散佈圖 (Scatter Plot)
    fig_scatter = px.scatter(
        filtered_df, 
        x="Cost_of_Living_USD", 
        y="Internet_Speed_Mbps",
        color="Region", 
        size="Safety_Score", 
        hover_name="City",
        text="City",
        trendline="ols", # 加上線性迴歸趨勢線，展現資料探勘的預測感
        title="城市生活成本與網路速度關聯（氣泡大小代表安全分數）",
        labels={"Cost_of_Living_USD": "每月生活成本 (USD)", "Internet_Speed_Mbps": "網路速度 (Mbps)"}
    )
    # 優化文字標籤顯示
    fig_scatter.update_traces(textposition='top center')
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# 頁面 3：🤖 機器學習：城市 AI 分群 (K-Means 演算法)
# ==========================================
elif page == "🤖 機器學習：城市 AI 分群":
    st.header("📌 演算法專區：K-Means 城市自動群聚分析")
    st.markdown("我們不使用官方排名，而是利用 **K-Means 機器學習演算法**，依據**生活成本、網速、安全、娛樂**四個維度，將全球城市自動分類。")
    
    # 讓使用者動態調整要分幾群 (K值)
    k_clusters = st.slider("請選擇 K-Means 的分群數量 (K 值)", min_value=2, max_value=5, value=3)
    
    # 特徵工程與標準化
    features = ['Cost_of_Living_USD', 'Internet_Speed_Mbps', 'Safety_Score', 'Entertainment_Score']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 執行 K-Means
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    df['Cluster'] = df['Cluster'].astype(str) # 轉成字串方便圖表分類
    
    # 用雷達圖或 3D 散佈圖展示分群結果
    st.subheader(f"🤖 演算法成功將城市劃分為 {k_clusters} 個族群")
    
    fig_cluster = px.scatter(
        df, 
        x="Cost_of_Living_USD", 
        y="Safety_Score",
        color="Cluster", 
        symbol="Cluster",
        size="Internet_Speed_Mbps",
        hover_name="City",
        title="K-Means 分群視覺化 (X軸: 成本, Y軸: 安全度, 氣泡大小: 網速)",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    # 顯示分群後的統計特徵，方便報告時解釋每一群的意義
    st.subheader("💡 探勘解密：各個群組的平均特徵")
    cluster_summary = df.groupby('Cluster')[features].mean().reset_index()
    st.dataframe(cluster_summary.style.format("{:.2f}", subset=features))

# ==========================================
# 頁面 4：🔍 頂級遊牧城市篩選器 (綜合指標)
# ==========================================
elif page == "🔍 頂級遊牧城市篩選器":
    st.header("📌 個人化智慧推薦：打造你的完美移居地")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_price = st.slider("你的最高每月預算 (USD)", min_value=800, max_value=4500, value=2500)
    with col2:
        min_internet = st.slider("最低網速要求 (Mbps)", min_value=20, max_value=500, value=100)
    with col3:
        min_safety = st.slider("最低安全度要求 (1-5 星)", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
        
    # 進行資料過濾
    recommend_df = df[
        (df['Cost_of_Living_USD'] <= max_price) & 
        (df['Internet_Speed_Mbps'] >= min_internet) & 
        (df['Safety_Score'] >= min_safety)
    ]
    
    st.subheader(f"🎉 根據您的嚴格條件，篩選出 {len(recommend_df)} 個符合的城市：")
    if not recommend_df.empty:
        st.dataframe(recommend_df.sort_values(by="Safety_Score", ascending=False))
        
        # 繪製加分雷達圖或條形圖
        fig_rec = px.bar(recommend_df, x="City", y="Internet_Speed_Mbps", color="Cost_of_Living_USD",
                         title="符合條件城市的網速與成本對比", color_continuous_scale="Viridis")
        st.plotly_chart(fig_rec, use_container_width=True)
    else:
        st.warning(" 糟糕！沒有城市同時滿足這麼嚴苛的條件，請試著放寬預算或網速標準！")