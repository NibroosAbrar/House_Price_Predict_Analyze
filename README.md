
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
## Data Preparation

Tahapan modeling bertujuan untuk membangun model machine learning yang dapat memprediksi harga rumah dengan akurat. Pada proyek ini, dua algoritma regresi digunakan:

- 🌲 **Random Forest Regressor**  
- 🚀 **XGBoost Regressor**  

Kedua model ini dipilih karena mampu menangani hubungan non-linear antar fitur, serta telah terbukti memiliki performa yang baik pada data real-world seperti properti dan valuasi harga.

---

### Split Data dan Scaling

Pertama, data dibagi menjadi **data latih (80%)** dan **data uji (20%)** menggunakan `train_test_split`. Selanjutnya, dilakukan **scaling** pada fitur numerik menggunakan `StandardScaler` untuk memastikan model seperti XGBoost dapat belajar secara optimal.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Model Development

Dua model machine learning yang digunakan dalam proyek ini:

---

#### 1. Random Forest Regressor

Random Forest adalah algoritma ensemble yang membangun beberapa pohon keputusan dan mengambil rata-rata prediksinya untuk meningkatkan akurasi dan mengurangi overfitting.

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```

- **Parameter utama**:
  - `n_estimators=100` → jumlah pohon yang dibangun
  - `random_state=42` → menjaga reprodusibilitas hasil

- **Kelebihan**:
  - Stabil dan tahan terhadap overfitting
  - Tidak memerlukan scaling
  - Dapat menampilkan feature importance

- **Kekurangan**:
  - Kurang optimal untuk dataset sangat besar
  - Interpretasi tidak sesederhana model linear

---

#### 2. XGBoost Regressor

XGBoost adalah algoritma boosting yang sangat powerful untuk task regresi, dengan performa tinggi dan banyak parameter yang bisa dituning untuk peningkatan performa.

```python
from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)
```

- **Parameter utama**:
  - `n_estimators=100` → jumlah boosting round
  - `learning_rate=0.1` → kecepatan belajar model baru dalam boosting

- **Kelebihan**:
  - Akurasi tinggi
  - Cocok untuk data kompleks dan banyak fitur
  - Mampu mengatasi overfitting

- **Kekurangan**:
  - Membutuhkan scaling
  - Kompleksitas dalam tuning parameter

---

## Evaluation

### Metrik Evaluasi yang Digunakan

Dalam proyek ini, kami menggunakan beberapa metrik evaluasi regresi untuk mengukur performa model dalam memprediksi nilai rumah:

- **R² Score (R-Squared)**  
  Mengukur seberapa baik variabel independen menjelaskan variabel dependen.  
  Formula:  
   $$
   R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
   $$ 
  Nilai R² berkisar antara 0 hingga 1, semakin mendekati 1 maka semakin baik performa model.

- **MAE (Mean Absolute Error)**  
  Rata-rata selisih absolut antara nilai prediksi dan nilai aktual.  
  Formula:  
  \[
  MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|
  \]  
  Metrik ini menunjukkan seberapa besar rata-rata kesalahan prediksi model dalam satuan yang sama dengan target.

- **RMSE (Root Mean Squared Error)**  
  Akar dari rata-rata kuadrat selisih antara nilai aktual dan nilai prediksi.  
  Formula:  
  \[
  RMSE = \sqrt{ \frac{1}{n} \sum (y_i - \hat{y}_i)^2 }
  \]  
  RMSE lebih sensitif terhadap outlier dibandingkan MAE karena menggunakan kuadrat selisih.

- **MAPE (Mean Absolute Percentage Error)**  
  Persentase rata-rata kesalahan prediksi terhadap nilai aktual.  
  Formula:  
  \[
  MAPE = \frac{1}{n} \sum \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%
  \]  
  Berguna untuk memahami kesalahan dalam konteks proporsi terhadap nilai aktual.

---

### Hasil Evaluasi Model

#### Random Forest Regressor

| Metrik | Nilai |
|--------|-------|
| R² Train | 0.9996 |
| R² Test  | 0.9993 |
| RMSE     | 1.12e+04 |
| MAE      | 8.95e+03 |
| MAPE     | 1.43% |

Model Random Forest menunjukkan performa yang sangat baik, dengan nilai R² yang tinggi baik pada data latih maupun data uji. MAE dan MAPE yang rendah menandakan bahwa model ini mampu memberikan prediksi yang sangat dekat dengan nilai aktual.

**Fitur Terpenting**:
- Square_Footage
- Year_Built
- Lot_Size

---

#### XGBoost Regressor

| Metrik | Nilai |
|--------|-------|
| R² Train | 0.9993 |
| R² Test  | 0.9992 |
| RMSE     | 1.23e+04 |
| MAE      | 9.25e+03 |
| MAPE     | 1.51% |

Model XGBoost juga menunjukkan performa yang sangat baik, dengan R² di atas 0.99. Namun, jika dibandingkan dengan Random Forest, nilai MAE dan MAPE sedikit lebih tinggi, menunjukkan bahwa prediksinya kurang presisi secara relatif.

**Fitur Terpenting**:
- Square_Footage
- Year_Built
- Lot_Size

---

### Model Terbaik

Berdasarkan evaluasi metrik:

- Random Forest memberikan hasil prediksi yang lebih akurat dibandingkan XGBoost.
- MAPE Random Forest lebih kecil (1.43%) dibanding XGBoost (1.51%), yang menunjukkan prediksi relatif lebih baik.
- Oleh karena itu, **Random Forest dipilih sebagai model terbaik** untuk proyek ini.

Model ini dapat digunakan secara praktis untuk memprediksi harga rumah dengan tingkat kesalahan yang sangat kecil dan stabil baik di data latih maupun uji.

## Kesimpulan

Berdasarkan proses analisis, pemodelan, dan evaluasi yang telah dilakukan, dapat diambil beberapa kesimpulan sebagai berikut:

1. **Pemilihan Model**  
   Dua algoritma telah dibandingkan, yaitu **Random Forest Regressor** dan **XGBoost Regressor**. Keduanya menunjukkan performa yang sangat baik, namun Random Forest memberikan hasil evaluasi yang lebih baik secara keseluruhan pada data uji.

2. **Kinerja Model**  
   - Model Random Forest mencapai nilai R² sebesar **0.9993** pada data uji, menunjukkan bahwa model mampu menjelaskan hampir seluruh variasi target.
   - Nilai MAPE Random Forest sebesar **1.43%**, lebih kecil dibanding XGBoost (**1.51%**), mengindikasikan tingkat kesalahan prediksi yang rendah dan konsisten.
   - RMSE dan MAE juga menunjukkan bahwa kesalahan absolut model sangat kecil.

3. **Fitur Paling Berpengaruh**  
   Baik Random Forest maupun XGBoost menunjukkan bahwa fitur **Square_Footage**, **Year_Built**, dan **Lot_Size** merupakan kontributor paling penting dalam memprediksi harga rumah.

4. **Rekomendasi**  
   Berdasarkan hasil evaluasi, **Random Forest** dipilih sebagai **model terbaik** karena:
   - Memberikan hasil yang lebih stabil dan akurat.
   - Lebih mudah untuk ditafsirkan dan digunakan dalam produksi.

5. **Kesesuaian Metrik**  
   Metrik yang digunakan seperti R², MAE, RMSE, dan MAPE telah sesuai dengan karakteristik data regresi dan problem statement yaitu meminimalkan kesalahan prediksi harga rumah.

---

Model yang dibangun telah mencapai performa yang sangat baik dan dapat diandalkan untuk memprediksi harga properti berdasarkan fitur-fitur penting. Langkah selanjutnya yang direkomendasikan adalah melakukan **deployment** model ini ke dalam aplikasi atau API yang bisa diakses oleh pengguna akhir untuk mendapatkan prediksi secara real-time.


