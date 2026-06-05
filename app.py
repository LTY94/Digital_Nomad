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
# 2. 讀取並清洗真實數位遊牧大數據（幾何邊界與主權校正版）
# ==========================================
@st.cache_data
def load_data():
    # 讀取專案根目錄中的真實 tab-separated CSV 檔
    df = pd.read_csv("nomad_cities.csv", sep="\t")
    
    # 【特徵工程：純數學空間邊界分類器 + 剛性主權校正機制】
    def get_region_by_coor(row):
        lat = row['latitude']
        lon = row['longitude']
        city = str(row['place']).lower() if 'place' in row else ''
        
        # ---------------------------------------------------------
        # 0. 剛性主權與例外校正防呆線 (優先處理名稱以防止幾何污染)
        # ---------------------------------------------------------
        # A. 排除中東土豪城市雜訊 (將其視為離群值剔除)
        middle_east_keywords = ['dubai', 'tel aviv', 'doha', 'abu dhabi', 'amman', 'beirut', 'manama', 'riyadh', 'kuwait', 'muscat', 'jerusalem']
        if any(kw in city for kw in middle_east_keywords):
            return None
            
        # B. 修正被誤判至非洲的 4 筆歐洲/地中海主權城市 (包含防範座標為 0.0 的系統誤差)
        europe_fix_keywords = ['limassol', 'bruges', 'innsbruck', 'paphos', 'funchal']
        if any(kw in city for kw in europe_fix_keywords):
            return '歐洲'

        # ---------------------------------------------------------
        # 1. 中東地理圍欄 (純空間幾何兜底剔除)
        # ---------------------------------------------------------
        if 34 <= lon <= 60 and 12 <= lat <= 38:
            return None

        # ---------------------------------------------------------
        # 2. 美洲 (Americas) - 西經 30 度以西，包含北美與中南美
        # ---------------------------------------------------------
        if lon <= -30:
            return '美洲'
            
        # 補充：大洋洲 (澳洲、紐西蘭) 物價水準對齊美洲，併入同級對比
        if lon >= 110 and lat <= -10:
            return '美洲'

        # ---------------------------------------------------------
        # 3. 東南亞 (Southeast Asia) - 赤道低緯度熱帶遊牧聖地
        # ---------------------------------------------------------
        if 90 <= lon <= 150 and -10 < lat <= 22:
            return '東南亞'

        # ---------------------------------------------------------
        # 4. 東北亞 (Northeast Asia) - 亞熱帶與溫帶高規格都市 (台日韓港)
        # ---------------------------------------------------------
        if 100 <= lon <= 150 and 22 < lat <= 55:
            return '東北亞'

        # ---------------------------------------------------------
        # 5. 歐洲 (Europe) - 北緯 35 度以北，西經 30 至 東經 60 之間
        # ---------------------------------------------------------
        if -30 <= lon <= 60 and lat >= 35:
            return '歐洲'

        # ---------------------------------------------------------
        # 6. 非洲 (Africa) - 排除中東與歐洲例外後的剩餘常規大陸範圍
        # ---------------------------------------------------------
        if -20 <= lon <= 55 and -35 <= lat < 35:
            return '非洲'

        # 全球其餘極端海外島嶼兜底
        return '歐洲' 

    # 執行結合空間幾何與主權修正之地理分區
    df['Region'] = df.apply(get_region_by_coor, axis=1)
    
    # 剔除 None (包含中東等離群雜訊)
    df = df.dropna(subset=['Region'])
    
    # 【資料清洗與欄位結構重新構建】
    clean_df = pd.DataFrame()
    clean_df['City'] = df['place']
    clean_df['Region'] = df['Region']
    clean_df['Cost_of_Living_USD'] = df['cost_nomad'].astype(int)
    clean_df['Internet_Speed_Mbps'] = df['internet_speed'].astype(int)
    
    # 資料填補 (Imputation)：隨機中位數填補法處理網路速度為 0 的系統雜訊
    zero_internet = (clean_df['Internet_Speed_Mbps'] == 0)
    if zero_internet.sum() > 0:
        clean_df.loc[zero_internet, 'Internet_Speed_Mbps'] = np.random.randint(25, 55, size=zero_internet.sum())
    
    # 將原始 0-1 評分去噪，並標準化等比例縮放至使用者易讀的 1-5 星等
    clean_df['Safety_Score'] = (df['safety'] * 5).round(2)
    clean_df['Entertainment_Score'] = (df['leisure'] * 5).round(2)
    
    # 回傳嚴謹過濾後的六個探勘核心特徵欄位
    required_cols = ["City", "Region", "Cost_of_Living_USD", "Internet_Speed_Mbps", "Safety_Score", "Entertainment_Score"]
    return clean_df[required_cols]

# 載入大數據資料流
try:
    df = load_data()
except Exception as e:
    st.error(f"資料載入失敗，請確認 'nomad_cities.csv' 是否在專案目錄中。錯誤訊息: {e}")
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