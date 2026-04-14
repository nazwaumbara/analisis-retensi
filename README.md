# 🛍️ Olist E-Commerce: Customer Intelligence & Revenue Growth Analysis

> **Portfolio Project — Junior Data Scientist**
> End-to-end analysis: EDA · RFM Segmentation · K-Means Clustering · Streamlit Dashboard

---

## 📌 Problem Statement

Olist, marketplace e-commerce terbesar di Brazil, menghadapi tantangan:
- **Tingginya churn rate** (>60% pelanggan tidak kembali setelah transaksi pertama)
- **Review score rendah** yang berkorelasi dengan penurunan repeat purchase
- **Revenue tidak merata** — sebagian kecil pelanggan menyumbang mayoritas revenue

**Pertanyaan Bisnis:**
1. Siapa pelanggan paling berharga Olist?
2. Apa penyebab utama pelanggan tidak kembali?
3. Apa rekomendasi aksi strategis yang bisa meningkatkan revenue?

---

## 🗂️ Dataset

**Source:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — Kaggle

| File | Deskripsi | Rows |
|------|-----------|------|
| `olist_orders_dataset.csv` | Master order data | ~100K |
| `olist_order_items_dataset.csv` | Item per order | ~112K |
| `olist_customers_dataset.csv` | Customer data | ~100K |
| `olist_order_payments_dataset.csv` | Payment records | ~104K |
| `olist_order_reviews_dataset.csv` | Review & score | ~100K |
| `olist_products_dataset.csv` | Product catalog | ~33K |

---

## 🔬 Methodology

```
Raw Data → Cleaning & Merging → EDA → RFM Analysis → K-Means Clustering → Insights → Dashboard
```

### RFM Framework
| Dimensi | Definisi | Bisnis Meaning |
|---------|----------|----------------|
| **Recency** | Hari sejak transaksi terakhir | Seberapa baru pelanggan aktif |
| **Frequency** | Jumlah transaksi unik | Seberapa sering pelanggan beli |
| **Monetary** | Total nilai pembelian | Seberapa besar kontribusi revenue |

---

## 📊 Key Findings

| Temuan | Data | Rekomendasi |
|--------|------|-------------|
| Champions hanya 8% pelanggan | Tapi sumbang 40%+ revenue | VIP Program |
| At Risk / Churned 25% basis | Inaktif >180 hari | Win-back campaign |
| Delay >5 hari → Score 1-2 | Korelasi kuat | Perbaikan SLA logistik |
| Health/Beauty = top revenue | R$980K | Prioritas inventory & ads |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Plotly](https://img.shields.io/badge/Plotly-5.17-purple)

---

## 🚀 How to Run

```bash
# 1. Clone repo
git clone https://github.com/username/olist-customer-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle & place in data/

# 4. Run analysis notebook
jupyter notebook olist_analysis.ipynb

# 5. Launch dashboard
streamlit run streamlit_dashboard.py
```

---

## 📁 Project Structure

```
olist-customer-intelligence/
├── data/                    # Raw CSVs from Kaggle (not committed)
├── output/                  # Generated charts
├── olist_analysis.py        # Main analysis script / notebook
├── streamlit_dashboard.py   # Interactive dashboard
├── requirements.txt
└── README.md
```

---

## 👤 Author

**[Namamu]** · Junior Data Scientist
[LinkedIn](#) · [Kaggle](#) · [Medium](#)
