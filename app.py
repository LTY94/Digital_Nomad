import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 網頁基本設定
# ==========================================
st.set_page_config(
    page_title="全球數位遊牧城市大數據探勘系統",
    page_icon="🌍",
    layout="wide"
)

# ==========================================
# 2. 讀取並清洗真實數位遊牧大數據
# ==========================================
@st.cache_data
def load_data():
    # 讀取使用者上傳的真實 tab-separated CSV 檔
    df = pd.read_csv("nomad_cities.csv", sep="\t")
    
    # 【特徵工程：利用經緯度進行地理智慧分區】
    def get_region_by_coor(row):
        lat = row['latitude']
        lon = row['longitude']
        if -170 <= lon <= -30: return '美洲'
        if -20 <= lon <= 55 and -35 <= lat <= 35:
            return '歐洲' if lat > 36 else '非洲'
        if -30 <= lon <= 45 and 35 <= lat <= 75: return '歐洲'
        if 110 <= lon <= 150 and 20 <= lat <= 50: return '東北亞'
        if 95 <= lon <= 145 and -11 <= lat <= 20: return '東南亞'
        if 95 <= lon <= 110 and 20 <= lat <= 25: return '東南亞'
        if lon > 60 and lon < 110: return '東南亞'
        if lon >= 110 and lon <= 160: return '東北亞'
        return '美洲' if (lon > 160 or lon < -170) else '歐洲'

    df['Region'] = df.apply(get_region_by_coor, axis=1)
    
    # 【資料清洗與欄位重構】
    df = df.rename(columns={
        'place': 'City',
        'cost_nomad': 'Cost_of_Living_USD',
        'internet_speed': 'Internet_Speed_Mbps'
    })
    
    # 資料填補 (Imputation)：將部分遺失或為 0 的網速填補為合理的隨機中位數
    df.loc[df['Internet_Speed_Mbps'] == 0, 'Internet_Speed_Mbps'] = np.random.randint(10, 30, size=(df['Internet_Speed_Mbps'] == 0).sum())
    
    # 將 0-1 的評分標準化縮放到 1-5 星等，方便使用者閱讀
    df['Safety_Score'] = (df['safety'] * 5).round(2)
    df['Entertainment_Score'] = (df['leisure'] * 5).round(2)
    
    # 只保留探勘核心欄位
    required_cols = ["City", "Region", "Cost_of_Living_USD", "Internet_Speed_Mbps", "Safety_Score", "Entertainment_Score"]
    return df[required_cols]

# 載入資料
try:
    df = load_data()
except Exception as e:
    st.error(f"資料載入失敗，請確保 'nomad_cities.csv' 位於專案根目錄中。錯誤訊息: {e}")
    st.stop()

# ==========================================
# 3. 側邊欄選單設計
# ==========================================
st.sidebar.title("🌍 導覽選單")
st.sidebar.markdown("請選擇您要查看的資料探勘維度：")
page = st.sidebar.radio(
    "分析面向",
    ["📊 城市基本數據分布", "💰 成本與網速關聯分析", "🤖 機器學習：城市 AI 分群", "🔍 頂級遊牧城市篩選器"]
)

st.title("🌍 全球數位遊牧（Digital Nomad）城市生活品質與成本綜合評估系統")
st.markdown("---")

# ==========================================
# 頁面 1：📊 城市基本數據分布
# ==========================================
if page == "📊 城市基本數據分布":
    st.header("📌 各大區域城市數據統計")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💡 各區域遊牧城市佔比 (圓餅圖)")
        region_counts = df['Region'].value_counts().reset_index()
        fig_pie = px.pie(region_counts, values='count', names='Region', hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.subheader("💰 各區域平均生活成本 (USD)")
        avg_cost = df.groupby('Region')['Cost_of_Living_USD'].mean().reset_index().sort_values(by='Cost_of_Living_USD')
        fig_bar = px.bar(avg_cost, x='Region', y='Cost_of_Living_USD', 
                         color='Cost_of_Living_USD', color_continuous_scale='Turbo',
                         labels={'Cost_of_Living_USD': '平均月花費 (USD)'})
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 原始探勘資料集摘要（真實 781 筆大數據）")
    st.dataframe(df.style.background_gradient(cmap='Blues', subset=['Cost_of_Living_USD', 'Internet_Speed_Mbps']))

# ==========================================
# 頁面 2：💰 成本與網速關聯分析
# ==========================================
elif page == "💰 成本與網速關聯分析":
    st.header("📌 尋找高 C/P 值天堂：網速 vs 生活成本")
    st.markdown("> **探勘目標：** 找出位於圖表**右下角**（生活成本低、網路速度極快）的夢幻遊牧城市。")
    
    selected_region = st.multiselect("篩選區域", options=df['Region'].unique(), default=df['Region'].unique())
    filtered_df = df[df['Region'].isin(selected_region)]
    
    fig_scatter = px.scatter(
        filtered_df, 
        x="Cost_of_Living_USD", 
        y="Internet_Speed_Mbps",
        color="Region", 
        size="Safety_Score", 
        hover_name="City",
        trendline="ols", # 加上進階線性迴歸趨勢線
        title="城市生活成本與網路速度關聯（氣泡大小代表安全分數，請將滑鼠移至點上查看城市名）",
        labels={"Cost_of_Living_USD": "每月生活成本 (USD)", "Internet_Speed_Mbps": "網路速度 (Mbps)"},
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# 頁面 3：🤖 機器學習：城市 AI 分群
# ==========================================
elif page == "🤖 機器學習：城市 AI 分群":
    st.header("📌 演算法專區：K-Means 城市自動群聚分析")
    st.markdown("利用 **K-Means 機器學習演算法**，依據**生活成本、網速、安全、娛樂**四個維度，將全球 781 座城市進行多維空間分群。")
    
    k_clusters = st.slider("請選擇 K-Means 的分群數量 (K 值)", min_value=2, max_value=5, value=3)
    
    # 特徵標準化處理
    features = ['Cost_of_Living_USD', 'Internet_Speed_Mbps', 'Safety_Score', 'Entertainment_Score']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 執行演算法
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    df['Cluster'] = "群組 " + df['Cluster'].astype(str)
    
    st.subheader(f"🤖 演算法成功將全球城市劃分為 {k_clusters} 個獨特特徵群組")
    
    fig_cluster = px.scatter(
        df, 
        x="Cost_of_Living_USD", 
        y="Safety_Score",
        color="Cluster", 
        symbol="Cluster",
        size="Internet_Speed_Mbps",
        hover_name="City",
        title="K-Means 多維分群投射圖 (X軸: 成本, Y軸: 安全度, 氣泡大小: 網速)",
        color_discrete_sequence=px.colors.qualitative.Dark2
    )
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    st.subheader("💡 探勘解密：各個群組的真實平均特徵矩陣")
    cluster_summary = df.groupby('Cluster')[features].mean().reset_index()
    st.dataframe(cluster_summary.style.format("{:.2f}", subset=features).background_gradient(cmap='YlOrRd'))

# ==========================================
# 頁面 4：🔍 頂級遊牧城市篩選器
# ==========================================
elif page == "🔍 頂級遊牧城市篩選器":
    st.header("📌 個人化智慧推薦：打造你的完美移居地")
    
    # 根據真實資料動態調整 Slider 上限
    max_cost_data = int(df['Cost_of_Living_USD'].max())
    max_internet_data = int(df['Internet_Speed_Mbps'].max())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_price = st.slider("你的最高每月預算 (USD)", min_value=500, max_value=max_cost_data, value=2500)
    with col2:
        min_internet = st.slider("最低網速要求 (Mbps)", min_value=5, max_value=max_internet_data, value=30)
    with col3:
        min_safety = st.slider("最低安全度要求 (1-5 星)", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
        
    recommend_df = df[
        (df['Cost_of_Living_USD'] <= max_price) & 
        (df['Internet_Speed_Mbps'] >= min_internet) & 
        (df['Safety_Score'] >= min_safety)
    ]
    
    st.subheader(f"🎉 根據您的嚴格條件，從小組資料庫篩選出 {len(recommend_df)} 個符合的真實城市：")
    if not recommend_df.empty:
        st.dataframe(recommend_df.sort_values(by="Safety_Score", ascending=False))
        
        fig_rec = px.bar(recommend_df.head(30), x="City", y="Internet_Speed_Mbps", color="Cost_of_Living_USD",
                         title="符合條件的前 30 個城市之網速與成本對比（顏色越偏藍代表成本越低）", color_continuous_scale="Blugrn")
        st.plotly_chart(fig_rec, use_container_width=True)
    else:
        st.warning(" 糟糕！沒有城市同時滿足這麼嚴苛的條件，請試著放寬預算或網速標準！")
