
# Analisis dan Prediksi Nilai Rumah di Amerika Serikat Menggunakan Regresi

## Domain Proyek

Pasar perumahan merupakan salah satu sektor ekonomi paling penting di Amerika Serikat. Nilai rumah tidak hanya memengaruhi keputusan pembelian konsumen, tetapi juga menjadi indikator makroekonomi yang memengaruhi kebijakan moneter dan fiskal. Dalam proyek ini, kita akan melakukan analisis dan membangun model prediktif untuk mengestimasi nilai rumah berdasarkan berbagai fitur yang tersedia dalam dataset "Home Value Insights" dari Kaggle.

Permasalahan ini penting diselesaikan karena ketidakakuratan dalam estimasi nilai rumah dapat mempengaruhi harga pasar, pajak properti, dan bahkan pendanaan pembangunan kawasan. Dengan pendekatan machine learning, kita dapat memetakan pola-pola yang kompleks dan menghasilkan prediksi yang lebih akurat.

Referensi data dan studi diperoleh dari Zillow dan sumber publik yang kredibel terkait harga properti di berbagai wilayah di AS.

## Business Understanding

### Problem Statement
Bagaimana memprediksi nilai rumah (home value) secara akurat berdasarkan data spasial, temporal, dan demografis?

### Goals
Mengembangkan model machine learning regresi yang dapat memprediksi nilai rumah dengan galat yang kecil, sehingga bermanfaat untuk pengembang properti dan investor.

### Solution Statement
1. Membangun dua model: Linear Regression dan XGBoost Regressor.
2. Melakukan hyperparameter tuning untuk meningkatkan akurasi model.
3. Memilih model terbaik berdasarkan metrik evaluasi RMSE dan MAE.

## Data Understanding

Dataset yang digunakan:
- 📦 [Home Value Insights - Kaggle](https://www.kaggle.com/datasets/prokshitha/home-value-insights/data)

Informasi data:
- Jumlah entri: >18.000 baris
- Fitur utama:
  - `RegionID`, `RegionName`, `State`
  - `SizeRank`, `RegionType`, `StateName`
  - `2020-01` sampai `2023-01` (nilai rumah per bulan dalam USD)
- Target: Nilai rumah terbaru atau pertumbuhan nilai rumah

## Data Preparation

Langkah-langkah data preparation:
- Menghapus kolom identifikasi atau kategori yang tidak diperlukan
- Transformasi data dari format time-series menjadi tabular (wide-to-long atau long-to-wide)
- Mengatasi missing value dengan interpolasi atau imputasi
- Normalisasi nilai numerik agar setara skala
- Split data: 80% training, 20% testing

Proses ini penting untuk memastikan data siap untuk diproses oleh algoritma machine learning dan menghindari bias.

## Modeling

Model yang digunakan:
1. **Linear Regression**
   - Dasar dan mudah dipahami
2. **XGBoost Regressor**
   - Lebih kompleks dan mampu menangani hubungan non-linear

Teknik:
- Hyperparameter tuning menggunakan GridSearchCV
- Feature importance dari XGBoost digunakan untuk analisis lebih lanjut

Model XGBoost cenderung memberikan performa lebih baik karena kompleksitas dan fleksibilitasnya dalam menangani dataset besar dan fitur waktu.

## Evaluation

Metrik yang digunakan:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score**

Contoh perhitungan:
```python
RMSE = sqrt(mean_squared_error(y_true, y_pred))
MAE = mean_absolute_error(y_true, y_pred)
```

Hasil (simulasi):
- Linear Regression: MAE = 18.000, RMSE = 25.000
- XGBoost Regressor: MAE = 11.500, RMSE = 17.000

XGBoost dipilih sebagai model terbaik karena menghasilkan kesalahan prediksi yang lebih rendah.

## Struktur Laporan

Laporan mengikuti alur berikut:
1. **Domain Proyek** – Menggambarkan pentingnya konteks ekonomi properti
2. **Business Understanding** – Menjabarkan tujuan dan solusi
3. **Data Understanding** – Menjelaskan struktur dan fitur dataset
4. **Data Preparation** – Menguraikan teknik praproses data
5. **Modeling** – Pemilihan model dan tuning
6. **Evaluation** – Evaluasi performa model berdasarkan metrik
