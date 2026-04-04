import pandas as pd

def clean_ecommerce_data(filepath):
    """
    Fungsi sistematis untuk membersihkan data mentah E-commerce secara otomatis.
    """
    # 1. Memuat Data Mentah
    # Dataset dari Kaggle ini biasanya dalam format Excel
    df = pd.read_excel(filepath, sheet_name='E Comm') 
    
    # 2. Standarisasi Kategori Teks
    # Seringkali ada data yang maknanya sama tapi tulisannya beda (typo dari sistem)
    df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace('Phone', 'Mobile Phone')
    df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace(['CC', 'COD'], ['Credit Card', 'Cash on Delivery'])
    
    # 3. Penanganan Missing Values (Imputasi Otomatis)
    # Daripada menghapus data, kita isi nilai yang kosong dengan pendekatan statistik
    
    # Isi nilai jarak gudang yang kosong dengan median (nilai tengah)
    df['WarehouseToHome'] = df['WarehouseToHome'].fillna(df['WarehouseToHome'].median())
    
    # Isi nilai durasi di aplikasi dengan rata-rata (mean)
    df['HourSpendOnApp'] = df['HourSpendOnApp'].fillna(df['HourSpendOnApp'].mean())
    
    # Isi kolom Tenure yang kosong dengan nilai 0 (asumsi pelanggan baru)
    df['Tenure'] = df['Tenure'].fillna(0)
    
    # 4. Membuang ID Pelanggan
    # CustomerID tidak memiliki nilai logis untuk prediksi (hanya ID unik), jadi kita buang agar tidak merusak model
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)
        
    # 5. Membersihkan Data Duplikat
    df = df.drop_duplicates()
    
    return df

# Eksekusi automasi:
# df_bersih = clean_ecommerce_data('E_Commerce_Dataset.xlsx')
# print(df_bersih.info())