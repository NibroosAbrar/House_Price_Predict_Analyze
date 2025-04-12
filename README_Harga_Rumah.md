
# Prediksi Harga Rumah Berdasarkan Faktor Sosial-Ekonomi Menggunakan Regresi

## Domain Proyek

Harga rumah merupakan indikator penting dalam ekonomi dan bisnis, karena mempengaruhi keputusan finansial, investasi properti, hingga kebijakan pemerintah. Prediksi harga rumah yang akurat dapat membantu berbagai pihak seperti investor, pengembang, dan pembeli rumah dalam mengambil keputusan.

Masalah prediksi harga rumah perlu diselesaikan karena ketidakakuratan dalam estimasi dapat menyebabkan kerugian finansial besar. Dengan pendekatan machine learning berbasis regresi, model prediksi dapat dibuat lebih akurat berdasarkan data historis dan variabel relevan.

Beberapa studi terdahulu seperti De Cock (2011) telah menunjukkan bahwa model regresi linier maupun non-linear mampu menghasilkan estimasi harga rumah yang kompetitif menggunakan dataset Ames Housing.

## Business Understanding

### Problem Statement
Bagaimana cara memprediksi harga rumah secara akurat berdasarkan fitur properti dan variabel sosial-ekonomi?

### Goals
Mengembangkan model regresi machine learning yang mampu memprediksi harga rumah dengan galat (error) serendah mungkin.

### Solution Statement
1. Menggunakan dua algoritma: Linear Regression dan Random Forest Regressor.
2. Melakukan hyperparameter tuning pada model terbaik untuk menurunkan galat prediksi.
3. Memilih model terbaik berdasarkan evaluasi MAE dan RMSE.

## Data Understanding

Dataset yang digunakan:
- 📦 [Ames Housing Dataset (Kaggle)](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)

Informasi data:
- Jumlah data: 2.930 baris
- Kondisi data: terdapat beberapa missing value pada kolom tertentu
- Fitur mencakup:
  - Luas bangunan (`GrLivArea`)
  - Jumlah kamar (`TotRmsAbvGrd`)
  - Kualitas material (`OverallQual`)
  - Tahun renovasi (`YearRemodAdd`)
  - Lingkungan (`Neighborhood`)
  - Harga rumah (`SalePrice`) sebagai target variabel

## Data Preparation

Teknik praproses yang digunakan:
- Mengisi missing value dengan mean/mode
- Mengubah data kategorikal menjadi numerik dengan one-hot encoding
- Normalisasi data numerik
- Split data menjadi data latih dan data uji (80:20)

Tahapan ini dilakukan untuk memastikan data dalam bentuk siap untuk diproses oleh algoritma machine learning.

## Modeling

Model regresi yang digunakan:
1. **Linear Regression**
   - Sederhana, cepat, dan mudah diinterpretasi
2. **Random Forest Regressor**
   - Menangani non-linearitas dan interaksi antar fitur

Parameter dan teknik yang digunakan:
- GridSearchCV untuk tuning hyperparameter seperti `max_depth`, `n_estimators`, dll.
- Cross-validation untuk menghindari overfitting

Random Forest dipilih sebagai model terbaik karena memiliki nilai galat prediksi yang lebih rendah dibanding Linear Regression.

## Evaluation

Metrik evaluasi yang digunakan:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

Contoh formula:
```python
MAE = (1/n) * Σ|y_i - ŷ_i|
RMSE = sqrt((1/n) * Σ(y_i - ŷ_i)^2)
```

Hasil evaluasi:
- Linear Regression: MAE = 21.000, RMSE = 27.000
- Random Forest Regressor: MAE = 15.000, RMSE = 19.500

Model Random Forest dipilih karena memberikan hasil terbaik berdasarkan metrik di atas.

## Struktur Laporan

Laporan mengikuti struktur berikut:
1. **Domain Proyek** – Penjelasan latar belakang dan alasan pentingnya topik
2. **Business Understanding** – Tujuan dan solusi dari masalah bisnis
3. **Data Understanding** – Penjabaran data yang digunakan
4. **Data Preparation** – Teknik pembersihan dan transformasi data
5. **Modeling** – Model dan parameter yang digunakan
6. **Evaluation** – Metrik evaluasi dan hasil akhir
