# -*- coding: utf-8 -*-
"""Projek Predictive Analysis_Nibroos.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jVGAFC51-U6-6VI-HHe91gfVBA_4Q_fR

# Mount Google Drive untuk membaca file CSV
### - Proses ini dilakukan untuk menghubungkan Google Colab dengan Google Drive Anda.
### - Dengan menghubungkan ke Google Drive, Anda dapat mengakses file dataset yang tersimpan di Drive.
"""

# Mount Google Drive untuk membaca file CSV
import os
from google.colab import drive
drive.mount('/content/drive')

loc = '/content/drive/My Drive/laskar ai/Projek Predictive Analysis'
os.chdir(loc)

os.getcwd()

"""# Install Package dan Library
### - Menginstal library XGBoost yang dibutuhkan untuk membangun model.
### - XGBoost adalah library gradient boosting yang populer untuk machine learning.
"""

!pip install xgboost

"""# Import Package dan Library
### - Mengimpor library yang dibutuhkan untuk analisis data dan machine learning.
### - Pandas dan NumPy untuk manipulasi data, Scikit-learn untuk model machine learning, Matplotlib dan Seaborn untuk visualisasi data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

"""# Load Data
### - Membaca dataset dari file CSV yang tersimpan di Google Drive.
### - Dataset akan dimuat ke dalam variabel 'df' sebagai Pandas DataFrame.
"""

file_path = '/content/drive/MyDrive/laskar ai/Projek Predictive Analysis/house_price_regression_dataset.csv'
df = pd.read_csv(file_path)

"""# Exploratory Data Analysis (EDA)
### - Menampilkan ukuran dataset (jumlah baris dan kolom).
### - Informasi ini penting untuk memahami dimensi data yang akan dianalisis.
### - Menampilkan tipe data dari setiap kolom.
### - Menampilkan 5 data teratas dari dataset untuk melihat sekilas struktur dan isi data.
### - Memeriksa missing values pada dataset. Missing values perlu ditangani sebelum membangun model.
### - Menampilkan statistik deskriptif dari dataset, seperti rata-rata, standar deviasi, dll., untuk mendapatkan gambaran tentang distribusi data.
### - Visualisasi korelasi antar fitur menggunakan heatmap untuk mengidentifikasi hubungan linear antar fitur.
### - Visualisasi distribusi fitur menggunakan histogram untuk menunjukkan frekuensi nilai untuk setiap fitur.
### - Visualisasi outlier menggunakan boxplot untuk mengidentifikasi data yang berada di luar rentang normal.
"""

print("Ukuran dataset:", df.shape)
print("\nTipe data:")
print(df.dtypes)

print("\n5 data teratas:")
print(df.head())

print("\nCek missing values:")
print(df.isnull().sum())

print("\nStatistik deskriptif:")
print(df.describe())

# Korelasi antar fitur
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Matriks Korelasi")
plt.show()

# Histogram untuk distribusi
df.hist(figsize=(12, 10), bins=30)
plt.suptitle("Distribusi Fitur")
plt.show()

# Boxplot untuk mendeteksi outlier
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Boxplot Fitur")
plt.show()

"""# Data Pre-Processing
### - Menghapus kolom yang tidak relevan (opsional).
### - Menangani missing values (jika ada) dengan metode yang sesuai, seperti menghapus baris dengan missing values atau menggantinya dengan nilai tertentu.
### - Memisahkan fitur dan target.
### - Melakukan One-Hot Encoding jika ada fitur kategorikal.
"""

# Hapus kolom yang tidak relevan (opsional)
# df = df.drop(columns=['id', 'nama'])

# Tangani missing values (jika ada)
df = df.dropna()  # Atau bisa juga df.fillna(df.mean())

# Pisahkan fitur dan target
# Ganti 'target' dengan nama kolom target kamu
X = df.drop(columns='House_Price')
y = df['House_Price']

# Cek apakah ada fitur kategorikal
categorical_columns = X.select_dtypes(include='object').columns
if len(categorical_columns) > 0:
    print("\nFitur kategorikal ditemukan:", categorical_columns)
    X = pd.get_dummies(X, columns=categorical_columns)

"""# Split Data dan Scaling
### - Membagi dataset menjadi data training dan data testing dengan rasio tertentu (misalnya, 80% untuk training dan 20% untuk testing).
### - Melakukan scaling fitur menggunakan StandardScaler untuk menyamakan skala data dan meningkatkan performa model.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""# Bangun Model
### - Membuat dan melatih model machine learning, seperti Random Forest dan XGBoost.
### - Menentukan hyperparameter model yang sesuai.
"""

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

"""# Evaluasi Model
### - Mengevaluasi performa model menggunakan metrik seperti R-squared, RMSE, MAE, dan MAPE.
### - Menganalisis hasil evaluasi untuk menentukan apakah model sudah optimal atau perlu dilakukan tuning lebih lanjut.
### - Menampilkan feature importance untuk melihat fitur yang paling berpengaruh terhadap prediksi model.
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model_full(name, model, X_train, X_test, y_train, y_test):
    # Prediksi
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Hitung metrik
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)

    # Cetak hasil
    print(f"\nModel: {name}")
    print("R² Train :", r2_train)
    print("R² Test  :", r2_test)
    print("RMSE     :", rmse_test)
    print("MAE      :", mae_test)
    print("MAPE (%) :", mape_test)

    # Evaluasi kondisi model
    if r2_train > 0.9 and r2_test < 0.7:
        print("⚠️  Model mengalami OVERFITTING.")
    elif r2_train < 0.5 and r2_test < 0.5:
        print("⚠️  Model kemungkinan UNDERFITTING.")
    elif abs(r2_train - r2_test) < 0.1:
        print("✅ Model FIT dengan baik (Balanced).")
    else:
        print("🔍 Model butuh tuning lebih lanjut.")

    # Feature Importance (jika tersedia)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X_train.columns if isinstance(X_train, pd.DataFrame) else [f'Feature {i}' for i in range(X_train.shape[1])]

        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 5))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.xlabel("Importance")
        plt.title(f"Feature Importance - {name}")
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

# Evaluasi Random Forest
evaluate_model_full("Random Forest", rf, X_train, X_test, y_train, y_test)

# Evaluasi XGBoost
evaluate_model_full("XGBoost", xgb, X_train, X_test, y_train, y_test)

"""Hasil: Model fit dengan baik dan akurasi, MAPE, MAE, dan RMSE sudah masuk dalam kategori yang sangat baik

# Save Model
### - Menyimpan model yang sudah dilatih ke dalam file agar dapat digunakan kembali di kemudian hari.
"""

import joblib

# Simpan model ke file .pkl
joblib.dump(rf, 'model_random_forest.pkl')
joblib.dump(xgb, 'model_xgboost.pkl')
print("✅ Model berhasil disimpan")

"""# Inferensi
### - Melakukan prediksi menggunakan model yang sudah dilatih.
### - Memberikan contoh input data dan menampilkan hasil prediksinya.
"""

def predict_from_input(model_name, input_dict):
    # Load model berdasarkan nama
    model_path = f"model_{model_name.lower()}.pkl"
    model = joblib.load(model_path)

    # Buat dataframe dari input
    input_df = pd.DataFrame([input_dict])

    # Prediksi
    prediction = model.predict(input_df)[0]
    print(f"📢 Prediksi dengan {model_name}: {prediction}")
    return prediction

# Contoh input (sesuai fitur yang kamu sebut)
sample_input = {
    'Square_Footage': 2000,
    'Num_Bedrooms': 3,
    'Num_Bathrooms': 2,
    'Year_Built': 2010,
    'Lot_Size': 0.5,
    'Garage_Size': 2,
    'Neighborhood_Quality': 8
}

# Prediksi
predict_from_input('xgboost', sample_input)