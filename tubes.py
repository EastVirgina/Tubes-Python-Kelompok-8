import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import math

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Mengatur gaya menggunakan Seaborn
sns.set_style('darkgrid')  

# Membaca data dari CSV
data = pd.read_csv("forestFires.csv")

# 1. Menampilkan Data Awal
print("=== Informasi Data ===")
print(data.info())
print("\n=== Statistik Deskriptif ===")
print(data.describe())


def plot_histograms(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols].hist(bins=30, figsize=(16,8), color='red', edgecolor='black')
    plt.tight_layout()
    plt.suptitle('Distribusi Variabel Numerik', fontsize=16, y=1.02)
    plt.show()

plot_histograms(data)

# 2. Heatmap Korelasi Antar Variabel (Hanya Kolom Numerik)
def plot_heatmap(df):
    plt.figure(figsize=(16,8))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap='Reds',
        fmt=".2f"
    )

    plt.title('Heatmap Korelasi Antar Variabel', fontsize=18)
    plt.show()

plot_heatmap(data)

# 3. barplot: Suhu vs Luas Kebakaran(top 5)
def plot_barplot_temp_area_top5(df):
    # Hitung luas total dan suhu rata-rata untuk setiap bulan.
    monthly_stats = df.groupby('month').agg({'area': 'sum', 'temp': 'mean'}).reset_index()
    
    # Urutkan bulan berdasarkan total luas wilayah dan ambil 5 bulan teratas.
    top5_months = monthly_stats.sort_values(by='area', ascending=False).head(5)['month']
    
    # Saring data untuk hanya menyertakan 5 bulan teratas.
    df_top5 = monthly_stats[monthly_stats['month'].isin(top5_months)]
    
    # Buatlah grafik batang untuk 5 bulan teratas.
    plt.figure(figsize=(16,8))
    sns.barplot(x='month', y='area', data=df_top5, color='red', alpha=0.7, label='Total Area')
    plt.title('Hubungan Suhu dengan Luas Kebakaran (Top 5)', fontsize=16)
    plt.xlabel('Bulan', fontsize=14)
    plt.ylabel('Luas Kebakaran (Total)', fontsize=14)

    # Tambahkan sumbu kedua untuk memplot suhu
    ax2 = plt.gca().twinx()
    sns.lineplot(x='month', y='temp', data=df_top5, color='orange', marker='o', ax=ax2, label='Temperature (°C)')
    ax2.set_ylabel('Suhu (°C)', fontsize=14)

    plt.show()

# Panggil fungsi untuk menampilkan barplot
plot_barplot_temp_area_top5(data)




# 4. countplot: Jumlah Kebakaran Berdasarkan Hari
def plot_count_day(df):
    plt.figure(figsize=(10,6))
    top = df.nlargest(50,'area')
    sns.countplot(x='day', data=top, hue='month',palette='Reds')
    plt.title('Jumlah Kebakaran Berdasarkan Hari', fontsize=16)
    plt.xlabel('Hari', fontsize=14)
    plt.ylabel('Jumlah Kebakaran', fontsize=14)
    plt.show()
    

# Panggil fungsi untuk menampilkan countplot
plot_count_day(data)

# 1. Visualisasi Distribusi Variabel Numerik
sns.set_style("darkgrid")

# Membaca data dari CSV
data = pd.read_csv("forestFires.csv")

# 2. MACHINE LEARNING (REGRESI)
# ===============================

# Encode kategori
data_ml = data.copy()
data_ml["month"] = data_ml["month"].astype("category").cat.codes
data_ml["day"] = data_ml["day"].astype("category").cat.codes

X = data_ml.drop("area", axis=1)
y = data_ml["area"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=== Evaluasi Model ===")
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2  :", r2_score(y_test, y_pred))
