# Food Recommendation Model

Repositori ini berisi proses riset, pengolahan data, dan pelatihan model untuk
**Sistem Rekomendasi Makanan** berbasis kemiripan nutrisi (*content-based filtering*).
Model dibangun di Jupyter Notebook lalu diekspor ke format **ONNX (Open Neural
Network eXchange)** agar dapat dijalankan lintas platform tanpa dependensi Python.

## Isi Repositori

| File | Deskripsi |
| --- | --- |
| `Food_Recommendation_System.ipynb` | Notebook utama: pra-pemrosesan data, pembangunan model, evaluasi, dan tracking. |
| `comprehensive_foods_usda.csv` | Dataset mentah USDA (~40.000 makanan, 24 kolom). |
| `healthy_foods_database.csv` | Dataset makanan sehat (~9.000 entri) sebagai penanda *healthy subset*. |
| `food_recommender.onnx` | Model rekomendasi hasil ekspor dalam format ONNX. |
| `foods.json` | Database makanan final (fitur nutrisi, alergen, dan diet) untuk inferensi. |
| `metadata.json` | Konfigurasi model: urutan fitur, alergen, kalori per tujuan, rasio makro, BMI, dan bobot skor. |
| `mlflow.db` | Database tracking eksperimen MLflow. |

## Alur Pembuatan Model

### 1. Pengumpulan & Penggabungan Data
- Dataset utama berasal dari **USDA** (`comprehensive_foods_usda.csv`) berisi ±40.000 makanan.
- Dataset `healthy_foods_database.csv` digunakan untuk menandai makanan mana yang
  tergolong "sehat" (`is_healthy_subset`) — terdapat ±7.572 nama makanan yang beririsan.

### 2. Pembersihan Data (*Data Cleaning*)
- **Deduplikasi**: menghapus makanan dengan nama duplikat (±9.100 baris).
- **Imputasi nilai kosong**: kolom nutrisi yang `NaN` diisi dengan **median per `food_type`**,
  sehingga imputasi tetap relevan terhadap kategori makanannya.
- **Koreksi nilai negatif**: beberapa nilai nutrisi (mis. `carbs_g` pada daging mentah)
  bernilai negatif dan di-*clip* ke minimal 0.

### 3. Rekayasa Fitur (*Feature Engineering*)
- **12 fitur nutrisi** dipakai sebagai dasar kemiripan:
  `calories`, `carbs_g`, `calcium_mg`, `fat_g`, `protein_g`, `saturated_fat_g`,
  `vitamin_c_mg`, `fiber_g`, `iron_mg`, `sodium_mg`, `sugar_g`, `cholesterol_mg`.
- **Deteksi alergen** berbasis *keyword matching* pada `ingredients` dan `food_name`
  untuk 6 kategori: `gluten`, `dairy`, `nuts`, `soy`, `eggs`, `fish`.
- **Klasifikasi diet** `is_vegetarian` dan `is_vegan` diturunkan dari `food_type`,
  kandungan daging/ikan, serta alergen terkait (dairy & eggs untuk vegan).

### 4. Pembangunan Model (*Content-Based Filtering*)
Model bekerja dengan menghitung **cosine similarity** antara vektor nutrisi target
pengguna dengan seluruh makanan di database:

1. **Normalisasi MinMax** seluruh fitur nutrisi ke rentang [0, 1].
2. **L2-normalization** tiap vektor makanan agar dot product setara cosine similarity.
3. Matriks fitur ternormalisasi di-*embed* langsung ke dalam graph ONNX sebagai konstanta.

Graph ONNX dibangun manual menggunakan `onnx.helper` dengan rangkaian operasi:
`Mul → Add` (MinMax scaling) → `ReduceL2 → Add(eps) → Div` (L2 normalize) →
`MatMul` (similarity terhadap seluruh makanan). Model menerima input
`target_nutrition` (12 fitur) dan menghasilkan `similarities` (skor untuk 30.900 makanan).

### 5. Ekspor & Validasi
- Model disimpan sebagai `food_recommender.onnx` (opset 18, IR version 9) dan
  divalidasi dengan `onnx.checker`.
- Database makanan diekspor ke `foods.json`, konfigurasi sistem ke `metadata.json`.

## Evaluasi Model

Evaluasi dilakukan dengan pendekatan **k-Nearest Neighbors** memanfaatkan skor
similarity yang dihasilkan model (makanan diri sendiri di-*exclude*).

**Precision@K** untuk konsistensi `food_type` dari rekomendasi teratas:

| K | Precision@K |
| :-: | :-: |
| 1 | 0.672 |
| 3 | 0.626 |
| 5 | 0.604 |

**Klasifikasi "Healthy"** (voting tetangga terdekat, `health_score >= 60`):

| Kelas | Precision | Recall | F1-score |
| --- | :-: | :-: | :-: |
| Not Healthy | 0.93 | 0.93 | 0.93 |
| Healthy | 0.78 | 0.80 | 0.79 |
| **Accuracy** | | | **0.90** |

## Experiment Tracking (MLflow)

Eksperimen dilacak menggunakan **MLflow** (`mlflow.db`), mencatat:
- **Parameter**: `K_neighbors`.
- **Metrik**: `accuracy`, `precision`.
- **Artefak**: model ONNX di-log via `mlflow.onnx.log_model`.

## Implementasi & Inferensi

Model dapat dijalankan dengan **ONNX Runtime** lintas bahasa/platform:

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("food_recommender.onnx")
input_name = sess.get_inputs()[0].name

# Vektor target nutrisi (12 fitur, sesuai feature_order pada metadata.json)
target = np.array([...], dtype=np.float32)

similarities = sess.run(None, {input_name: target})[0]
top5 = np.argsort(-similarities)[:5]   # indeks makanan paling mirip
```

Indeks hasil rekomendasi merujuk ke entri pada `foods.json`. Skor akhir rekomendasi
dapat dikombinasikan dengan `health_score` dan bonus *healthy subset* sesuai bobot pada
`scoring_weights` di `metadata.json` (similarity 0.6, health_score 0.3, healthy bonus 0.1).

## Dependensi

```bash
pip install onnx onnxruntime mlflow pandas numpy scikit-learn
```
