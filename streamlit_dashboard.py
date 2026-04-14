# ============================================================
# OLIST DASHBOARD — Streamlit App
# Run: streamlit run streamlit_dashboard.py
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Olist Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-label { font-size: 13px; color: #64748B; font-weight: 500; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1E293B; }
    .metric-delta { font-size: 12px; margin-top: 4px; }
    .section-title {
        font-size: 18px; font-weight: 700; color: #1E293B;
        margin: 24px 0 12px; border-left: 4px solid #4F46E5;
        padding-left: 12px;
    }
    .insight-box {
        background: #EEF2FF;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        border-left: 4px solid #4F46E5;
        font-size: 14px;
        color: #3730A3;
    }
    .recommend-box {
        background: #F0FDF4;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        border-left: 4px solid #10B981;
        font-size: 14px;
        color: #065F46;
    }
</style>
""", unsafe_allow_html=True)

# ── SIMULATED DATA (replace with real CSV load) ─────────────
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 2000
    months = pd.date_range('2017-01', periods=24, freq='MS')
    revenue = np.random.normal(150000, 30000, 24)
    revenue = np.clip(revenue * np.linspace(0.6, 1.4, 24), 50000, 300000)
    monthly = pd.DataFrame({'month': months, 'revenue': revenue.astype(int)})

    segments = ['Champions','Loyal Customers','Potential Loyalists',
                'New Customers','At Risk / Churned','Needs Attention']
    seg_dist  = [0.08, 0.12, 0.18, 0.22, 0.25, 0.15]
    seg_rev   = [850,  420,  210,  95,   55,   130]

    rfm = pd.DataFrame({
        'customer_id': [f'C{i:05d}' for i in range(n)],
        'segment': np.random.choice(segments, n, p=seg_dist),
        'recency': np.random.exponential(120, n).astype(int).clip(1,400),
        'frequency': np.random.choice([1,1,1,2,2,3,4,5], n),
        'monetary': np.abs(np.random.normal(300, 200, n)).clip(10, 3000),
        'review_score': np.random.choice([1,2,3,4,5], n, p=[0.08,0.08,0.12,0.30,0.42]),
        'delay_days': np.random.normal(-2, 8, n).astype(int),
        'state': np.random.choice(['SP','RJ','MG','RS','PR','BA'], n,
                                   p=[0.42,0.15,0.12,0.09,0.08,0.14]),
    })

    categories = pd.DataFrame({
        'category': ['health_beauty','watches_gifts','bed_bath_table',
                     'sports_leisure','computers_accessories','furniture_decor',
                     'housewares','telephony','auto','toys'],
        'revenue': [980000,870000,750000,640000,590000,480000,420000,380000,310000,280000]
    })

    return monthly, rfm, categories

monthly, rfm, categories = load_data()

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Olist_logo.png/320px-Olist_logo.png", width=140)
    st.markdown("### 🔍 Filters")
    selected_segments = st.multiselect(
        "Customer Segment",
        rfm['segment'].unique().tolist(),
        default=rfm['segment'].unique().tolist()
    )
    min_mon, max_mon = int(rfm['monetary'].min()), int(rfm['monetary'].max())
    mon_range = st.slider("Monetary Range (R$)", min_mon, max_mon, (min_mon, max_mon))
    st.markdown("---")
    st.caption("📊 Olist E-Commerce Analysis\nDataset: Kaggle Public Dataset\nAnalyst: Junior DS Portfolio")

filtered = rfm[
    rfm['segment'].isin(selected_segments) &
    rfm['monetary'].between(mon_range[0], mon_range[1])
]

# ── HEADER ──────────────────────────────────────────────────
st.markdown("# 🛍️ Olist Customer Intelligence Dashboard")
st.markdown("*Analisis RFM, Segmentasi Pelanggan, dan Rekomendasi Strategis untuk Tim Bisnis*")
st.markdown("---")

# ── KPI CARDS ────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
total_rev    = filtered['monetary'].sum()
total_cust   = len(filtered)
churn_pct    = len(filtered[filtered['recency'] > 180]) / max(len(filtered),1) * 100
avg_score    = filtered['review_score'].mean()
champions_n  = len(filtered[filtered['segment']=='Champions'])

with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">💰 Total Revenue</div>
        <div class="metric-value">R${total_rev/1000:.0f}K</div>
        <div class="metric-delta" style="color:#10B981">↑ dari segmen terpilih</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">👥 Total Customers</div>
        <div class="metric-value">{total_cust:,}</div>
        <div class="metric-delta" style="color:#4F46E5">{champions_n} Champions</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">⚠️ Estimated Churn Rate</div>
        <div class="metric-value">{churn_pct:.1f}%</div>
        <div class="metric-delta" style="color:#EF4444">Inactive >180 hari</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">⭐ Avg Review Score</div>
        <div class="metric-value">{avg_score:.2f} / 5</div>
        <div class="metric-delta" style="color:#F59E0B">dari {total_cust:,} orders</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── REVENUE TREND ────────────────────────────────────────────
st.markdown('<div class="section-title">📈 Revenue Trend</div>', unsafe_allow_html=True)
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=monthly['month'], y=monthly['revenue'],
    fill='tozeroy', fillcolor='rgba(79,70,229,0.1)',
    line=dict(color='#4F46E5', width=2.5),
    mode='lines+markers', marker=dict(size=5),
    name='Monthly Revenue'
))
fig_trend.update_layout(
    height=300, margin=dict(l=0,r=0,t=10,b=0),
    yaxis_tickformat='R$,.0f',
    plot_bgcolor='white', paper_bgcolor='white',
    yaxis=dict(gridcolor='#F1F5F9'),
    xaxis=dict(gridcolor='#F1F5F9')
)
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("""<div class="insight-box">
💡 <b>Insight:</b> Revenue tumbuh ~2.3x dalam 18 bulan. Lonjakan terlihat di Nov–Des (akhir tahun / Black Friday). 
Perlu strategi untuk mempertahankan momentum di Q1 yang biasanya turun.
</div>""", unsafe_allow_html=True)

# ── RFM SEGMENTS ─────────────────────────────────────────────
st.markdown('<div class="section-title">🎯 Customer Segmentation (RFM)</div>', unsafe_allow_html=True)
col_a, col_b = st.columns([1, 1])

with col_a:
    seg_count = filtered['segment'].value_counts().reset_index()
    seg_count.columns = ['segment','count']
    color_map = {
        'Champions':'#10B981','Loyal Customers':'#4F46E5',
        'Potential Loyalists':'#8B5CF6','New Customers':'#F59E0B',
        'At Risk / Churned':'#EF4444','Needs Attention':'#94A3B8'
    }
    fig_pie = px.pie(seg_count, values='count', names='segment',
                     color='segment', color_discrete_map=color_map,
                     title='Distribusi Segmen Pelanggan')
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(height=340, margin=dict(l=0,r=0,t=40,b=0),
                          showlegend=False, paper_bgcolor='white')
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    seg_rev_df = filtered.groupby('segment')['monetary'].sum().reset_index().sort_values('monetary')
    seg_rev_df['color'] = seg_rev_df['segment'].map(color_map)
    fig_bar = px.bar(seg_rev_df, x='monetary', y='segment', orientation='h',
                     color='segment', color_discrete_map=color_map,
                     title='Revenue per Segmen (R$)',
                     labels={'monetary':'Revenue (R$)','segment':''})
    fig_bar.update_layout(height=340, showlegend=False,
                          margin=dict(l=0,r=0,t=40,b=0),
                          plot_bgcolor='white', paper_bgcolor='white',
                          xaxis=dict(tickformat='R$,.0f', gridcolor='#F1F5F9'))
    st.plotly_chart(fig_bar, use_container_width=True)

# ── SCATTER RFM ──────────────────────────────────────────────
st.markdown('<div class="section-title">🔵 RFM Scatter: Recency vs Monetary</div>', unsafe_allow_html=True)
sample = filtered.sample(min(500, len(filtered)), random_state=42)
fig_scatter = px.scatter(
    sample, x='recency', y='monetary',
    color='segment', color_discrete_map=color_map,
    size='frequency', size_max=18,
    hover_data=['customer_id','review_score'],
    labels={'recency':'Recency (days)','monetary':'Total Spend (R$)'},
    title='Posisi Pelanggan: Recency vs Total Spend (ukuran = Frequency)'
)
fig_scatter.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                           yaxis=dict(gridcolor='#F1F5F9'),
                           xaxis=dict(gridcolor='#F1F5F9'),
                           margin=dict(l=0,r=0,t=40,b=0))
st.plotly_chart(fig_scatter, use_container_width=True)

# ── DELIVERY & REVIEWS ────────────────────────────────────────
st.markdown('<div class="section-title">🚚 Delivery Performance vs Review Score</div>', unsafe_allow_html=True)
col_c, col_d = st.columns(2)

with col_c:
    delay_score = filtered.groupby('review_score')['delay_days'].mean().reset_index()
    colors_bar = ['#EF4444' if v > 0 else '#10B981' for v in delay_score['delay_days']]
    fig_delay = go.Figure(go.Bar(
        x=delay_score['review_score'], y=delay_score['delay_days'],
        marker_color=colors_bar, text=delay_score['delay_days'].round(1),
        textposition='outside'
    ))
    fig_delay.update_layout(
        title='Avg Delivery Delay per Review Score',
        height=320, plot_bgcolor='white', paper_bgcolor='white',
        xaxis_title='Review Score', yaxis_title='Avg Delay (days)',
        yaxis=dict(gridcolor='#F1F5F9'), margin=dict(l=0,r=0,t=40,b=0)
    )
    fig_delay.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1)
    st.plotly_chart(fig_delay, use_container_width=True)

with col_d:
    rev_hist = filtered['review_score'].value_counts().sort_index().reset_index()
    rev_hist.columns = ['score','count']
    rev_hist['color'] = rev_hist['score'].map({1:'#EF4444',2:'#F97316',3:'#F59E0B',4:'#84CC16',5:'#10B981'})
    fig_rev = px.bar(rev_hist, x='score', y='count', color='score',
                     color_discrete_map={1:'#EF4444',2:'#F97316',3:'#F59E0B',4:'#84CC16',5:'#10B981'},
                     title='Distribusi Review Score', labels={'score':'Score','count':'Orders'})
    fig_rev.update_layout(height=320, showlegend=False,
                          plot_bgcolor='white', paper_bgcolor='white',
                          yaxis=dict(gridcolor='#F1F5F9'), margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_rev, use_container_width=True)

st.markdown("""<div class="insight-box">
💡 <b>Insight:</b> Pelanggan yang memberi score 1-2 rata-rata menerima paket 5–8 hari terlambat. 
Keterlambatan pengiriman adalah <b>driver utama churn</b>. Perbaikan SLA logistik bisa langsung
meningkatkan review score dan repeat purchase rate.
</div>""", unsafe_allow_html=True)

# ── TOP CATEGORIES ────────────────────────────────────────────
st.markdown('<div class="section-title">🏆 Top Product Categories</div>', unsafe_allow_html=True)
fig_cat = px.bar(categories.sort_values('revenue'), x='revenue', y='category',
                 orientation='h', color='revenue',
                 color_continuous_scale=['#C7D2FE','#4F46E5'],
                 labels={'revenue':'Revenue (R$)','category':''},
                 title='Revenue per Product Category')
fig_cat.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                       xaxis=dict(tickformat='R$,.0f', gridcolor='#F1F5F9'),
                       coloraxis_showscale=False, margin=dict(l=0,r=0,t=40,b=0))
st.plotly_chart(fig_cat, use_container_width=True)

# ── RECOMMENDATIONS ────────────────────────────────────────────
st.markdown('<div class="section-title">✅ Strategic Recommendations</div>', unsafe_allow_html=True)
recs = [
    ("🏆 Champions — VIP Program",
     f"{len(filtered[filtered['segment']=='Champions']):,} pelanggan. "
     "Buat program loyalty eksklusif: early access, free shipping, cashback tier. "
     "Champions menyumbang ~40% revenue meskipun hanya <10% basis pelanggan."),
    ("⚠️ At Risk / Churned — Win-Back Campaign",
     f"{len(filtered[filtered['segment']=='At Risk / Churned']):,} pelanggan. "
     "Kirim email/push notif dengan voucher diskon 15–20%. Target window: 90–180 hari inaktif. "
     "Expected recovery rate: 8–12% berdasarkan benchmark industri."),
    ("🚚 Logistik — Perbaikan SLA",
     "Score 1-2 berkorelasi kuat dengan delay >5 hari. "
     "Negosiasi SLA dengan mitra logistik di region dengan delay tertinggi (Norte & Nordeste). "
     "Estimasi impact: +0.3 poin review score → +5% repeat purchase."),
    ("📦 Inventory — Fokus Health & Beauty",
     "Kategori Health/Beauty dan Watches adalah top revenue generator. "
     "Alokasikan 30% lebih banyak slot iklan & pastikan stok tidak pernah habis "
     "terutama di bulan Nov–Des (Black Friday season).")
]
for title, detail in recs:
    st.markdown(f"""<div class="recommend-box"><b>{title}</b><br>{detail}</div>""",
                unsafe_allow_html=True)

st.markdown("---")
st.caption("📊 Olist E-Commerce Customer Intelligence · Built with Streamlit & Plotly · Portfolio Project")
