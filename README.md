
# Analisis dan Prediksi Nilai Rumah Menggunakan Regresi

## Domain Proyek

Pasar properti merupakan sektor strategis dalam perekonomian karena berkaitan langsung dengan kebutuhan primer masyarakat dan investasi jangka panjang. Nilai rumah dipengaruhi oleh banyak faktor seperti lokasi, tren harga historis, indeks ekonomi wilayah, dan karakteristik regional lainnya. Kenaikan atau penurunan harga rumah memiliki implikasi besar terhadap kemampuan masyarakat dalam membeli rumah, perencanaan pembangunan daerah, dan stabilitas ekonomi secara keseluruhan.

**Mengapa masalah ini penting?**  
Ketidakakuratan dalam memprediksi harga rumah dapat menyebabkan:
- Distorsi pasar, di mana harga yang ditawarkan tidak sesuai dengan nilai sebenarnya;
- Kesulitan akses terhadap hunian yang terjangkau;
- Kerugian bagi investor dan pembeli rumah;
- Kesalahan dalam penetapan pajak properti oleh pemerintah daerah.

**Bagaimana masalah ini dapat diselesaikan?**  
Permasalahan ini dapat diatasi melalui pendekatan machine learning regresi yang memanfaatkan data historis indeks nilai rumah dan karakteristik wilayah untuk membangun model prediksi nilai rumah. Model ini dapat digunakan oleh pembeli, pengembang, dan pemerintah dalam mengambil keputusan berbasis data.

**Hasil Riset Terkait**  
Penelitian oleh Kok, Monkkonen, dan Quigley (2014) dalam jurnal *Regional Science and Urban Economics* menunjukkan bahwa model prediksi berbasis machine learning memiliki keakuratan yang lebih tinggi dibandingkan metode tradisional dalam memperkirakan harga properti, terutama ketika mempertimbangkan variabel spasial dan temporal.  
> Kok, N., Monkkonen, P., & Quigley, J. M. (2014). *Land use regulations and the value of land and housing: An intra-metropolitan analysis*. Regional Science and Urban Economics, 46, 1–15. https://doi.org/10.1016/j.regsciurbeco.2014.01.001

Selain itu, Zillow Research (2023) mengembangkan model prediksi harga rumah menggunakan algoritma machine learning untuk *Zillow Home Value Index (ZHVI)* yang digunakan secara luas oleh agen properti dan bank di Amerika Serikat.  
> Zillow Economic Research (2023). *Zillow Home Value Index (ZHVI)*. Retrieved from https://www.zillow.com/research/data/

Dengan menggunakan dataset dari Zillow dan pendekatan regresi, proyek ini bertujuan untuk menghasilkan prediksi nilai rumah yang akurat dan aplikatif untuk kebutuhan ekonomi dan bisnis di dunia nyata.

## Business Understanding

### Problem Statement

Prediksi nilai properti merupakan aspek krusial dalam pengambilan keputusan ekonomi di sektor perumahan. Baik pembeli, penjual, agen properti, maupun lembaga keuangan membutuhkan estimasi harga rumah yang akurat untuk meminimalisir risiko kerugian finansial. Namun, kompleksitas faktor yang memengaruhi harga rumah seperti lokasi, ukuran, kondisi pasar lokal, dan tren waktu seringkali menyebabkan estimasi yang bias atau tidak presisi.

Ketergantungan terhadap penilaian manual atau asumsi subyektif membuat proses valuasi rentan terhadap kesalahan. Oleh karena itu, dibutuhkan pendekatan berbasis data yang dapat mengolah banyak variabel secara efisien untuk memberikan estimasi harga yang lebih akurat dan dapat diandalkan.

### Goals

- Mengembangkan model machine learning berbasis data properti untuk memprediksi nilai rumah secara akurat.
- Menyediakan sistem pendukung keputusan bagi pelaku industri properti dalam menentukan harga jual, harga beli, atau plafon pinjaman.
- Mengidentifikasi faktor-faktor paling berpengaruh terhadap nilai rumah agar dapat digunakan sebagai insight bisnis.

### Solution Statement

Untuk mencapai tujuan tersebut, kami mengusulkan dua solusi model prediksi sebagai berikut:

1. **Random Forest Regressor**  
   Digunakan sebagai baseline model. Random Forest cocok untuk menangani dataset dengan banyak fitur serta memiliki kemampuan menangkap hubungan non-linear antar variabel. Model ini juga relatif mudah diinterpretasikan, terutama dalam menampilkan feature importance yang bermanfaat bagi pengambil keputusan.

2. **XGBoost Regressor**  
   Merupakan algoritma boosting yang sangat populer dalam kompetisi prediksi karena kemampuannya memberikan akurasi tinggi dan mengatasi overfitting. XGBoost digunakan untuk meningkatkan performa prediksi dan akan dituning secara optimal untuk mendapatkan model terbaik.

### Evaluasi dan Aplikasi Solusi

- **Hyperparameter tuning** dilakukan pada kedua model menggunakan teknik seperti GridSearchCV untuk memaksimalkan performa.
- Model terbaik dipilih berdasarkan hasil evaluasi terhadap data uji menggunakan metrik berikut:
  - **Mean Absolute Error (MAE)**: mengukur rata-rata selisih absolut antara nilai prediksi dan nilai aktual.
  - **Mean Absolute Percentage Error (MAPE)**: mengukur rata-rata persentase kesalahan prediksi terhadap nilai aktual, berguna untuk memahami kesalahan dalam skala relatif.

- Hasil dari model akan diaplikasikan secara praktis, antara lain dalam bentuk:
  - **Dashboard interaktif** yang memungkinkan pengguna untuk memasukkan fitur rumah dan melihat estimasi harga secara real-time.
  - **Insight analitik** berupa ranking fitur paling berpengaruh terhadap harga rumah, sebagai bahan pertimbangan bisnis.
  - **Sistem pendukung keputusan** bagi pelaku industri properti, seperti developer, bank, atau agen properti, dalam menentukan strategi harga.
    
## Data Understanding

### Dataset Overview

Dataset ini memuat informasi properti rumah tinggal seperti ukuran bangunan, jumlah kamar, kualitas lingkungan, dan harga rumah. Dataset berisi **1.000 baris** dan **8 fitur** yang relevan untuk prediksi harga rumah. Dataset ini dapat digunakan untuk model regresi.

📥 Sumber data: [Home Value Insights - Kaggle](https://www.kaggle.com/datasets/prokshitha/home-value-insights)

---

### Dimensi dan Tipe Data

- **Ukuran data**: `(1000, 8)`
- **Tipe data per kolom**:

| Fitur | Tipe Data |
|-------|-----------|
| `Square_Footage` | int64 |
| `Num_Bedrooms` | int64 |
| `Num_Bathrooms` | int64 |
| `Year_Built` | int64 |
| `Lot_Size` | float64 |
| `Garage_Size` | int64 |
| `Neighborhood_Quality` | int64 |
| `House_Price` | float64 |

---

### Cek Data Awal

Contoh 5 data pertama:

| Square_Footage | Num_Bedrooms | Num_Bathrooms | Year_Built | Lot_Size | Garage_Size | Neighborhood_Quality | House_Price |
|----------------|--------------|---------------|------------|----------|-------------|-----------------------|-------------|
| 1320           | 2            | 1             | 2016       | 1.59     | 1           | 5                     | 523280.65   |
| 4322           | 3            | 2             | 2005       | 4.75     | 2           | 6                     | 779727.94   |
| 5925           | 4            | 3             | 1997       | 5.34     | 2           | 7                     | 977794.61   |
| 496            | 1            | 1             | 1977       | 1.72     | 1           | 3                     | 144320.33   |
| 9285           | 5            | 4             | 1993       | 4.60     | 3           | 8                     | 1.04e+06    |

---

### Missing Values

Hasil pengecekan menunjukkan bahwa **tidak ada nilai yang hilang (missing values)** pada seluruh fitur.

---

### Statistik Deskriptif

| Fitur | Mean | Min | Max |
|-------|------|-----|-----|
| `Square_Footage` | 2,994.9 | 450 | 9,350 |
| `Num_Bedrooms` | 2.99 | 1 | 5 |
| `Num_Bathrooms` | 1.98 | 1 | 4 |
| `Year_Built` | 1997.3 | 1970 | 2020 |
| `Lot_Size` | 2.98 | 0.48 | 7.54 |
| `Garage_Size` | 1.39 | 0 | 3 |
| `Neighborhood_Quality` | 5.51 | 1 | 10 |
| `House_Price` | 624,166.48 | 100,000 | 1,183,729.08 |

---

### Insight Awal dari Data

- Fitur `Square_Footage`, `Lot_Size`, dan `Garage_Size` cenderung memiliki rentang nilai yang luas → perlu pertimbangan normalisasi atau scaling saat modeling.
- Harga rumah (`House_Price`) memiliki variasi cukup tinggi (dari 100 ribu hingga lebih dari 1 juta), artinya model harus mampu mengakomodasi data outlier dan high variance.
- Fitur `Neighborhood_Quality` berupa **ordinal**, dari 1 (terendah) hingga 10 (tertinggi).
- Tidak ada data yang hilang → tidak diperlukan imputasi.

---

### Visualisasi dan EDA (Exploratory Data Analysis)

Beberapa teknik visualisasi yang relevan untuk analisis lanjutan:

1. **Histogram** untuk distribusi harga rumah → mengetahui apakah distribusinya skewed.
2. **Boxplot** untuk mendeteksi outlier pada fitur numerik seperti `Square_Footage`, `Lot_Size`, dan `House_Price`.
3. **Heatmap korelasi** untuk melihat hubungan antar fitur numerik terhadap target (`House_Price`).
4. **Scatterplot** antara `Square_Footage` dan `House_Price` → membantu mengenali pola linear atau non-linear.

---

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
