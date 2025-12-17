import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import math
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

# 3. Heatmap Korelasi Antar Variabel (Hanya Kolom Numerik)
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

# 4. barplot: Suhu vs Luas Kebakaran(top 5)
def plot_barplot_temp_area_top5(df):
    # Calculate the total area and average temperature for each month
    monthly_stats = df.groupby('month').agg({'area': 'sum', 'temp': 'mean'}).reset_index()
    
    # Sort the months by total area and get the top 5 months
    top5_months = monthly_stats.sort_values(by='area', ascending=False).head(5)['month']
    
    # Filter the data to include only the top 5 months
    df_top5 = monthly_stats[monthly_stats['month'].isin(top5_months)]
    
    # Plot the barplot for the top 5 months
    plt.figure(figsize=(16,8))
    sns.barplot(x='month', y='area', data=df_top5, color='red', alpha=0.7, label='Total Area')
    plt.title('Hubungan Suhu dengan Luas Kebakaran (Top 5)', fontsize=16)
    plt.xlabel('Bulan', fontsize=14)
    plt.ylabel('Luas Kebakaran (Total)', fontsize=14)

    # Add a second axis to plot the temperature
    ax2 = plt.gca().twinx()
    sns.lineplot(x='month', y='temp', data=df_top5, color='orange', marker='o', ax=ax2, label='Temperature (°C)')
    ax2.set_ylabel('Suhu (°C)', fontsize=14)

    plt.show()

# Example usage
plot_barplot_temp_area_top5(data)


# 5. barplot: Luas Kebakaran top 3 
def plot_top3_barplot_area_month(df):
    # Calculate the total area for each month
    monthly_area = df.groupby('month')['area'].sum().reset_index()
    
    # Sort the months by total area and get the top 3 months
    top3_months = monthly_area.sort_values(by='area', ascending=False).head(3)['month']
    
    # Filter the data to include only the top 3 months
    df_top3 = df[df['month'].isin(top3_months)]
    
    # Plot the barplot for the top 3 months (showing the average area for each month)
    plt.figure(figsize=(16,8))
    sns.barplot(x='month', y='area', data=df_top3, hue='month', palette='Reds')
    plt.title('Distribusi Luas Kebakaran per Bulan (Top 3)', fontsize=16)
    plt.xlabel('Bulan', fontsize=14)
    plt.ylabel('Luas Kebakaran (Rata-rata)', fontsize=14)
    plt.show()

# Example usage
plot_top3_barplot_area_month(data)

# 6. Count Plot: Jumlah Kebakaran Berdasarkan Hari
def plot_count_day(df):
    plt.figure(figsize=(10,6))
    top = df.nlargest(50,'area')
    sns.countplot(x='day', data=top, hue='month',palette='Reds')
    plt.title('Jumlah Kebakaran Berdasarkan Hari', fontsize=16)
    plt.xlabel('Hari', fontsize=14)
    plt.ylabel('Jumlah Kebakaran', fontsize=14)
    plt.show()
    

plot_count_day(data)

# 7. Pair Plot: Hubungan Antar Variabel Numerik
from matplotlib import rcParams
def plot_pairplot(df):
    rcParams['figure.figsize'] = 16,8
    sns.pairplot(df.select_dtypes(include=[np.number])[['temp', 'RH', 'wind', 'rain', 'area']], diag_kind='kde', corner=True)
    plt.suptitle('Pair Plot Hubungan Antar Variabel Numerik', fontsize=16, y=1.02)
    plt.show()

plot_pairplot(data)

# 8. Scatter Plot Lokasi (X, Y) dengan Warna berdasarkan Luas Kebakaran
def plot_scatter_location(df):
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(df['X'], df['Y'], c=df['area'], cmap='RdYlBu', alpha=0.6)
    plt.colorbar(scatter, label='Luas Kebakaran')
    plt.title('Sebaran Kebakaran berdasarkan Lokasi (I, V)', fontsize=16)
    plt.xlabel('I', fontsize=14)
    plt.ylabel('V', fontsize=14)
    plt.show()

plot_scatter_location(data)

def plot_trend_kebakaran(df):
    df_sorted = df.copy()
    # Mengonversi bulan menjadi angka untuk mengurutkan
    month_order = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    df_sorted['month_num'] = df_sorted['month'].map(month_order)
    # Mengurutkan berdasarkan 'month_num' dan 'day'
    day_order = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 'sat':6, 'sun':7}
    df_sorted['day_num'] = df_sorted['day'].map(day_order)
    df_sorted = df_sorted.sort_values(by=['month_num', 'day_num']).reset_index(drop=True)
    df_sorted['cumulative_area'] = df_sorted['area'].cumsum()
    
    plt.figure(figsize=(12,6))
    plt.plot(df_sorted.index, df_sorted['cumulative_area'], color='red')
    plt.title('Trend Cumulative Luas Kebakaran dari Waktu ke Waktu', fontsize=16)
    plt.xlabel('Urutan Waktu', fontsize=14)
    plt.ylabel('Cumulative Luas Kebakaran', fontsize=14)
    plt.show()

plot_trend_kebakaran(data)

# 10. Visualisasi Interaktif dengan Plotly (top 5)

def plot_interactive_bar_top5(df):
    try:
        # Select the top 5 rows based on the 'area' column
        top5_df = df.nlargest(5, 'area')

        # Create a bar plot with the filtered DataFrame
        fig = px.bar(top5_df, x='temp', y='area', color='month',
                     title='Top 5 Hubungan Suhu dengan Luas Kebakaran (Interaktif) - Bar Plot',
                     labels={'temp': 'Suhu (°C)', 'area': 'Luas Kebakaran'},
                     hover_data=['X', 'Y', 'wind', 'rain'])
        fig.update_layout(title_font_size=20)  # Change title font size
        fig.show()
    except ValueError as e:
        print("Terjadi kesalahan saat menampilkan Plotly Bar Plot:", e)
        print("Pastikan bahwa paket 'nbformat' telah terinstal dan versi Plotly serta nbformat memenuhi persyaratan.")

# Call the function with your data
plot_interactive_bar_top5(data)

# 11. **Menampilkan Nilai yang Hilang**
def data_miss (df):
        print("\n=== Mengecek Nilai yang Hilang ===")
        plt.figure(figsize=(15,5))
        null = (df.isnull().sum())
        plt.bar(null, color='Red', height=null)
        plt.title("MISS DATA")
data_miss(data)

#12. Hujan di Suatu Area

def hujan_area_top5(df):
    # Select the top 5 rows based on the 'area' column
    top5_df = df.nlargest(5, 'area')
    
    # Create the scatter plot with the filtered DataFrame
    plt.figure(figsize=(10, 6))
    hujan = top5_df['rain']
    hari = top5_df['day']
    wilayah = top5_df['area']
    
    # Plot the scatter plot
    plt.scatter(hari, wilayah, c=hujan, cmap='Reds')
    
    # Add color bar and label
    hujan = plt.colorbar(orientation="horizontal")
    hujan.set_label(label="Hujan", size=18)
    
    # Set labels and title
    plt.xlabel("Hari", fontsize=18)
    plt.ylabel("Area", fontsize=18)
    plt.title("Top 5 Hujan di Setiap Area", fontsize=18)
    
    # Display the plot
    plt.show()

# Call the function with your data
hujan_area_top5(data)

#13.Kecepatan angin di setiap bulanya (top 5)
def plot_wind_month_top5(df):
    # Select the top 5 rows based on the 'wind' column
    top5_df = df.nlargest(5, 'wind')
    
    # Create a barplot using sns.catplot or sns.barplot (based on your preference)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='wind', y='day', data=top5_df, hue='month', palette="Reds")
    
    # Title and other plot customizations
    plt.title("Top 5 Wind Speed by Month", fontsize=16)
    plt.xlabel("Wind Speed", fontsize=14)
    plt.ylabel("Day", fontsize=14)
    plt.legend(title="Month", fontsize=12)
    plt.show()

# Call the function with your data
plot_wind_month_top5(data)

#14. Suhu di setiap bulan
def temp_bulan (df):
        plt.figure(figsize=(10,6))
        top5 = df.nlargest(6,'temp')
        sns.lineplot(x=data.day.sort_values(ascending=False), y=data.temp.sort_values(ascending=False), data=top5, hue='month', palette="Reds")
        plt.title("Suhu di Setiap Hari", fontsize = 18)
        
        plt.show()

temp_bulan(data)

#15.Kelembapan Saat Hujan
def rh_hujan (df) :
        plt.figure(figsize=(10,6))
        top = df.nlargest(50, 'RH')
        sns.barplot(x='rain', y='RH' , data=top, hue='month', palette="Reds")
        plt.xlabel("Hujan 0.0 mm (tidak ada hujan) hingga 6.4 mm (curah hujan maksimum yang tercatat)", fontsize=12)
        plt.ylabel("Kelembapan Udara(rentang: (15 - 100)%", fontsize=12)
        plt.title("Kelembapan saat hujan", fontsize=18)
        
        ax2 = plt.gca().twinx()
        sns.lineplot(x='rain', y='day', data=top, color='orange', marker='o', ax=ax2, label='Month')
        ax2.set_ylabel('Day', fontsize=14)
                
        plt.show()
        
rh_hujan(data)

#16. Potensi Terjadi Kebakaran
from matplotlib import rcParams
def potensi_kebakaran (df) :
        rcParams['figure.figsize'] = 16,8
        top = df.nlargest(150,'FFMC')
        sns.barplot(x='day', y='FFMC' , data=top, hue='month', palette="Reds")
        plt.xlabel("Hari", fontsize=18)
        plt.ylabel("FFMC (18.7 = sangat basah, 96.20 = sangat kering)", fontsize=12)
        plt.title("Top 7 Potensi Kemudahan Terjadi Kebakaran (FFMC)", fontsize=12)
        plt.show()
        
potensi_kebakaran(data)

#17. Potensi Penyebaran api
def panas_temperatur(df) :
        plt.figure(figsize=(18,8))
        top10_df = df.nlargest(3, 'ISI')
        brun = df['ISI']
        temperatur = df['temp']
        hari = df['day']
        sns.barplot(x='day', y='ISI', data=top10_df, hue='month', palette='Reds')
        plt.title(" potensi penyebaran awal api.", fontsize=18)
        plt.ylabel(" 0.0 (tidak menyebar) hingga 56.10 (sangat cepat menyebar).", fontsize=14)
        plt.xlabel("mon - sun", fontsize=18)
        plt.show()
        
panas_temperatur(data)

# 18. Kekeringan Dalam Wilayah
from matplotlib import rcParams
def dc_area (df) :
        rcParams['figure.figsize'] = 16, 8
        top10_df = df.nlargest(20, 'DC')
        sns.displot(x='area', y='DC', data=top10_df, hue='day', palette='Reds')
        plt.title("Kekeringan Dalam Wilayah")
        plt.xlabel("Area rentang 0.00 (tidak ada kebakaran) hingga 1090.84 (kebakaran besar).", size=10)
        plt.ylabel("KBB rentang 7.9 (sangat basah) hingga 860.6 (sangat kering)", size=10)
        
        plt.show()
        
dc_area(data)

#19. Kelembapan saat Kekurangan Bahan Bakar Ringan
from matplotlib import rcParams
def angin_kelembapan (df):
        rcParams['figure.figsize'] = 10,8
        top_hari = df.nlargest(15, 'FFMC')
        hari = top_hari['day']
        kbbr = top_hari['FFMC']
        kelembapan = top_hari['RH']
        plt.scatter(hari, kbbr, c=kelembapan, cmap='Reds', edgecolors='k')
        bar = plt.colorbar(orientation='horizontal')
        bar.set_label("Kelembapan", fontsize=16)
        plt.title("Kelembapan saat Kekurangan Bahan Bakar Ringan", fontsize=16)
        plt.xlabel("Day Mon-Sun", fontsize=12)
        plt.ylabel(" kelembapan bahan bakar ringan, seperti dedaunan dan ranting\n  18.7 (basah) hingga 96.20 (kering).")
        plt.show()
                
        
angin_kelembapan(data)

#20. Suhu saat Hujan untuk Setiap Bulan
from matplotlib import rcParams
def xy_kebakaran (df) :
        rcParams['figure.figsize'] = 10,5
        top10_df = df.nlargest(8, 'rain')
        plt.title("Suhu saat Hujan untuk Setiap Bulan", fontsize=18)
        sns.barplot(x=data.rain.sort_values(ascending=False), y='temp', data=top10_df, hue='month', palette='Reds')
        plt.xlabel("Kecepatan Hujan\n 0.0 (tidak ada hujan) hingga 6.4 (hujan lebat)", fontsize=14)
        plt.ylabel("Suhu \n (2.2°C hingga 33.30°C.)", fontsize=14)
xy_kebakaran(data)
plt.show()

        
        

