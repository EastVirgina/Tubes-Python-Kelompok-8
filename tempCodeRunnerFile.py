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

