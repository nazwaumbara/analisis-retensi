# ============================================================
# OLIST E-COMMERCE ANALYSIS
# Junior Data Scientist Portfolio Project
# ============================================================

# %% [markdown]
# # 🛍️ Olist E-Commerce: Customer Intelligence & Revenue Growth
# **Objective:** Identify who our most valuable customers are, why customers churn,
# and what strategic actions can increase revenue.
#
# **Dataset:** Brazilian E-Commerce Public Dataset by Olist (Kaggle)
# **Analyst:** [Your Name] | [Date]

# %% SECTION 1: SETUP & IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Style config
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
PALETTE = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
sns.set_palette(PALETTE)

print("✅ Libraries loaded. Ready to analyze.")

# %% SECTION 2: LOAD DATA
# Download from: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# Place all CSVs in a folder called 'data/'

orders       = pd.read_csv('data/olist_orders_dataset.csv')
order_items  = pd.read_csv('data/olist_order_items_dataset.csv')
customers    = pd.read_csv('data/olist_customers_dataset.csv')
payments     = pd.read_csv('data/olist_order_payments_dataset.csv')
reviews      = pd.read_csv('data/olist_order_reviews_dataset.csv')
products     = pd.read_csv('data/olist_products_dataset.csv')
category_map = pd.read_csv('data/product_category_name_translation.csv')

print(f"Orders: {orders.shape} | Items: {order_items.shape} | Customers: {customers.shape}")

# %% SECTION 3: DATA CLEANING & MERGING
# Parse dates
date_cols = ['order_purchase_timestamp','order_delivered_customer_date',
             'order_estimated_delivery_date']
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col])

# Filter: only delivered orders
orders_clean = orders[orders['order_status'] == 'delivered'].copy()

# Master dataframe
df = (orders_clean
      .merge(order_items[['order_id','price','freight_value']], on='order_id', how='left')
      .merge(customers[['customer_id','customer_state']], on='customer_id', how='left')
      .merge(payments.groupby('order_id')['payment_value'].sum().reset_index(), on='order_id', how='left')
      .merge(reviews[['order_id','review_score']].drop_duplicates('order_id'), on='order_id', how='left')
)

# Delivery delay feature
df['delivery_delay_days'] = (
    df['order_delivered_customer_date'] - df['order_estimated_delivery_date']
).dt.days

# Revenue per order
df['total_order_value'] = df['price'] + df['freight_value']

print(f"✅ Master dataframe: {df.shape}")
print(df.head(3))

# %% SECTION 4: EXPLORATORY DATA ANALYSIS

# --- 4.1 Revenue Trend Over Time ---
df['month'] = df['order_purchase_timestamp'].dt.to_period('M')
monthly_revenue = df.groupby('month')['total_order_value'].sum().reset_index()
monthly_revenue['month_str'] = monthly_revenue['month'].astype(str)

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(range(len(monthly_revenue)), monthly_revenue['total_order_value'],
                alpha=0.15, color=PALETTE[0])
ax.plot(range(len(monthly_revenue)), monthly_revenue['total_order_value'],
        color=PALETTE[0], linewidth=2.5, marker='o', markersize=4)
ax.set_xticks(range(0, len(monthly_revenue), 3))
ax.set_xticklabels(monthly_revenue['month_str'][::3], rotation=45, ha='right', fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x/1000:.0f}K'))
ax.set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Month')
ax.set_ylabel('Total Revenue (BRL)')
plt.tight_layout()
plt.savefig('output/01_revenue_trend.png', dpi=150, bbox_inches='tight')
plt.show()

# 💡 BUSINESS INSIGHT: Tandai bulan dengan lonjakan revenue & catat konteksnya

# --- 4.2 Top Product Categories ---
products_eng = products.merge(category_map, on='product_category_name', how='left')
items_cat = order_items.merge(products_eng[['product_id','product_category_name_english']],
                               on='product_id', how='left')
top_categories = (items_cat.groupby('product_category_name_english')['price']
                  .sum().sort_values(ascending=False).head(10))

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.barh(top_categories.index[::-1], top_categories.values[::-1], color=PALETTE[0])
ax.bar_label(bars, labels=[f'R${v/1000:.0f}K' for v in top_categories.values[::-1]],
             padding=5, fontsize=9)
ax.set_title('Top 10 Product Categories by Revenue', fontsize=14, fontweight='bold')
ax.set_xlabel('Total Revenue (BRL)')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x/1000:.0f}K'))
plt.tight_layout()
plt.savefig('output/02_top_categories.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 4.3 Review Score Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
score_counts = df['review_score'].value_counts().sort_index()
axes[0].bar(score_counts.index, score_counts.values,
            color=[PALETTE[3] if s <= 2 else PALETTE[1] if s == 5 else PALETTE[2] for s in score_counts.index])
axes[0].set_title('Review Score Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Score')
axes[0].set_ylabel('Number of Orders')

# Delay vs Score
delay_score = df.groupby('review_score')['delivery_delay_days'].mean()
axes[1].bar(delay_score.index, delay_score.values,
            color=[PALETTE[3] if v > 0 else PALETTE[1] for v in delay_score.values])
axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[1].set_title('Avg Delivery Delay by Review Score', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Review Score')
axes[1].set_ylabel('Avg Delay (days)')
plt.tight_layout()
plt.savefig('output/03_reviews_delay.png', dpi=150, bbox_inches='tight')
plt.show()

# 💡 BUSINESS INSIGHT: Keterlambatan sangat berkorelasi dengan review buruk

# %% SECTION 5: RFM ANALYSIS & CUSTOMER SEGMENTATION
# RFM = Recency, Frequency, Monetary — framework standar industri

snapshot_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

rfm = (df.groupby('customer_id')
       .agg(
           recency   = ('order_purchase_timestamp', lambda x: (snapshot_date - x.max()).days),
           frequency = ('order_id', 'nunique'),
           monetary  = ('total_order_value', 'sum')
       ).reset_index())

print(f"\n📊 RFM Summary:")
print(rfm[['recency','frequency','monetary']].describe().round(2))

# Score each dimension 1–4
rfm['r_score'] = pd.qcut(rfm['recency'],   q=4, labels=[4,3,2,1]).astype(int)
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=4, labels=[1,2,3,4]).astype(int)
rfm['m_score'] = pd.qcut(rfm['monetary'],  q=4, labels=[1,2,3,4]).astype(int)
rfm['rfm_score'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']

# Segment labels
def assign_segment(row):
    if row['rfm_score'] >= 10:
        return 'Champions'
    elif row['rfm_score'] >= 8:
        return 'Loyal Customers'
    elif row['rfm_score'] >= 6:
        return 'Potential Loyalists'
    elif row['r_score'] >= 3:
        return 'New Customers'
    elif row['rfm_score'] <= 4:
        return 'At Risk / Churned'
    else:
        return 'Needs Attention'

rfm['segment'] = rfm.apply(assign_segment, axis=1)

seg_summary = (rfm.groupby('segment')
               .agg(customers=('customer_id','count'),
                    avg_monetary=('monetary','mean'),
                    avg_recency=('recency','mean'))
               .sort_values('avg_monetary', ascending=False))

print("\n📊 Segment Summary:")
print(seg_summary.round(2))

# Visualise segments
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors_seg = {'Champions': PALETTE[1], 'Loyal Customers': PALETTE[0],
              'Potential Loyalists': PALETTE[4], 'New Customers': PALETTE[2],
              'At Risk / Churned': PALETTE[3], 'Needs Attention': '#94A3B8'}
seg_counts = rfm['segment'].value_counts()

wedges, texts, autotexts = axes[0].pie(
    seg_counts, labels=seg_counts.index, autopct='%1.1f%%',
    colors=[colors_seg.get(s, '#ccc') for s in seg_counts.index],
    startangle=140, pctdistance=0.75)
for at in autotexts: at.set_fontsize(9)
axes[0].set_title('Customer Segment Distribution', fontsize=13, fontweight='bold')

seg_rev = rfm.groupby('segment')['monetary'].sum().sort_values(ascending=True)
bars = axes[1].barh(seg_rev.index, seg_rev.values,
                    color=[colors_seg.get(s, '#ccc') for s in seg_rev.index])
axes[1].bar_label(bars, labels=[f'R${v/1000:.0f}K' for v in seg_rev.values], padding=5, fontsize=9)
axes[1].set_title('Total Revenue by Segment', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Total Revenue (BRL)')
plt.tight_layout()
plt.savefig('output/04_rfm_segments.png', dpi=150, bbox_inches='tight')
plt.show()

# %% SECTION 6: K-MEANS CLUSTERING (ML)
# Validasi segmentasi RFM dengan unsupervised learning

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['recency','frequency','monetary']])

# Elbow method
inertias = []
sil_scores = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(rfm_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(K_range, inertias, marker='o', color=PALETTE[0])
axes[0].set_title('Elbow Method', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')

axes[1].plot(K_range, sil_scores, marker='s', color=PALETTE[1])
axes[1].set_title('Silhouette Score per K', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
plt.tight_layout()
plt.savefig('output/05_elbow.png', dpi=150, bbox_inches='tight')
plt.show()

# Fit optimal K (misal 4)
OPTIMAL_K = 4
km_final = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
rfm['cluster'] = km_final.fit_predict(rfm_scaled)

cluster_profile = rfm.groupby('cluster')[['recency','frequency','monetary']].mean().round(2)
print("\n📊 Cluster Profiles:")
print(cluster_profile)

# %% SECTION 7: KEY BUSINESS INSIGHTS SUMMARY
champions     = rfm[rfm['segment']=='Champions']
at_risk       = rfm[rfm['segment']=='At Risk / Churned']
churn_rate    = len(rfm[rfm['recency'] > 180]) / len(rfm) * 100
avg_ltv       = rfm['monetary'].mean()

print("\n" + "="*55)
print("       📋 KEY BUSINESS INSIGHTS SUMMARY")
print("="*55)
print(f"  Total unique customers    : {len(rfm):,}")
print(f"  Champions (top tier)      : {len(champions):,} ({len(champions)/len(rfm)*100:.1f}%)")
print(f"  At Risk / Churned         : {len(at_risk):,} ({len(at_risk)/len(rfm)*100:.1f}%)")
print(f"  Estimated churn rate      : {churn_rate:.1f}% (inactive >180 days)")
print(f"  Average Customer LTV      : R${avg_ltv:.2f}")
print(f"  Champions avg spend       : R${champions['monetary'].mean():.2f}")
print("="*55)
print("""
  RECOMMENDATIONS:
  1. Champions   → VIP loyalty program + early access promos
  2. At Risk      → Win-back email campaign with discount voucher
  3. Late delivery→ SLA improvement; delivery delay = #1 churn driver
  4. Top category → Increase inventory & ads for Health/Beauty & Watches
""")

print("✅ Analysis complete! Check /output folder for charts.")
