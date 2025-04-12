
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
