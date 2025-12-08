# Klasifikasi Penyakit Daun Kopi Menggunakan Metode Machine Learning Klasik, Deep Learning dan Visual Transformer

# Coffee Leaf Disease Classification

Klasifikasi Penyakit Daun Kopi dengan **Machine Learning Klasik, Deep Learning (CNN), dan Visual Transformer (ViT + LoRA + XAI)**

Proyek ini membangun sistem klasifikasi citra daun kopi menjadi **3 kelas**:

* `Daun_Sehat`
* `Daun_Bercak` (*Cercospora coffeicola*)
* `Daun_Karat` (*Hemileia vastatrix*)

Pendekatan yang digunakan:

1. **ML Klasik**

   * Ekstraksi fitur warna & tekstur (HSV, GLCM, LBP).
   * Training beberapa model (LR, KNN, RF, SVM, MLP).
   * **Weighted Voting Ensemble** + teknik perbaikan lanjutan.

2. **Deep Learning (CNN)**

   * CNN dari awal (no transfer learning).
   * Augmentasi data dengan `ImageDataGenerator`.

3. **Visual Transformer (ViT) + LoRA + XAI**

   * Fine-tuning **Vision Transformer (ViT)** pre-trained (`google/vit-base-patch16-224-in21k`) untuk klasifikasi daun kopi.
   * **LoRA (Low-Rank Adaptation)** untuk fine-tuning yang lebih ringan dan efisien.
   * **Explainable AI (XAI)** dengan **Grad-CAM** dan penjelasan tekstual tambahan via **Groq LLM**.

---

## 1. Dataset

* Dataset citra daun kopi **seimbang** (balanced) untuk 3 kelas.

* Disimpan di Google Drive dan di-split menggunakan library `split-folders` dengan rasio:

  * **Train:** 80%
  * **Validation:** 10%
  * **Test:** 10%

* Struktur folder akhir:

  ```text
  dataset_coffee_split/
  ├── train/
  │   ├── Daun_Sehat/
  │   ├── Daun_Bercak/
  │   └── Daun_Karat/
  ├── val/
  │   ├── Daun_Sehat/
  │   ├── Daun_Bercak/
  │   └── Daun_Karat/
  └── test/
      ├── Daun_Sehat/
      ├── Daun_Bercak/
      └── Daun_Karat/
  ```

Script split data menggunakan:

```python
splitfolders.ratio(
    original_dataset_path,
    output=output_split_path,
    seed=42,
    ratio=(0.8, 0.1, 0.1),
    move=False
)
```

---

## 2. Pipeline Machine Learning Klasik

### 2.1. Load & Preprocessing Gambar

* Baca citra dengan **OpenCV**.
* Resize ke `128×128`.
* Normalisasi piksel ke rentang `[0, 1]`.
* **Augmentasi manual** (untuk semua kelas):

  * Rotasi random (±15°)
  * Flip horizontal / vertical
  * Penyesuaian brightness & contrast
  * Zoom in/out ringan

Sehingga jumlah citra training bertambah dan distribusi kelas tetap seimbang.

### 2.2. Ekstraksi Fitur

Dari setiap gambar diekstrak **56 fitur** utama:

1. **Histogram HSV**

   * 8 bin untuk Hue, 8 untuk Saturation, 8 untuk Value → **24 fitur**.

2. **GLCM (Gray-Level Co-occurrence Matrix)**

   * Grayscale direduksi ke 64 level, jarak = 1, sudut = 0°, 45°, 90°, 135°.
   * Diambil rata-rata properti:
     `contrast, dissimilarity, homogeneity, energy, correlation, ASM`
     → **6 fitur**.

3. **LBP (Local Binary Patterns)**

   * Radius = 3, `n_points = 8 * radius = 24`, method `"uniform"`.
   * Histogram LBP (n_points + 2) → **26 fitur**.

Total: **24 + 6 + 26 = 56 fitur per gambar.**

### 2.3. Preprocessing Fitur

* **StandardScaler** pada seluruh fitur.
* Opsional: **PCA** (di kode disediakan, default dimatikan).
* **SelectKBest (f_classif)** untuk memilih **25 fitur terbaik**.

### 2.4. Model yang Dilatih

Menggunakan `GridSearchCV` + `StratifiedKFold (5-fold)`:

* **Support Vector Machine (SVM)**
* **Random Forest**
* **K-Nearest Neighbors (KNN)**
* **Logistic Regression**
* **Multi-Layer Perceptron (MLP)**

Setiap model:

* Di-tuning hyperparameter (C, gamma, n_estimators, dst).
* Dipilih model terbaik berdasarkan **F1-weighted** di validation set.

### 2.5. Ensemble & Perbaikan

Setelah model individu:

1. **Weighted Voting Ensemble**

   * Menggabungkan semua `best_models`.
   * Bobot tiap model = F1-weighted di validation.

2. **Confidence Threshold**

   * Jika confidence < threshold (mis. 0.7), prediksi diganti kelas default (kelas mayoritas).

3. **Model Calibration**

   * `CalibratedClassifierCV` dengan metode `sigmoid` & `isotonic`.

4. **Stacking Ensemble**

   * Base estimators = semua `best_models`.
   * Meta-learner = Logistic Regression.

5. **Error Analysis Lengkap**

   * Confusion matrix (raw & normalized).
   * Analisis pasangan kesalahan (True → Predicted).
   * Visualisasi gambar yang salah klasifikasi.
   * Analisis error rate per kelas dan rekomendasi augmentasi lanjutan.

### 2.6. Ringkasan Hasil (ML Klasik)

* Model-model klasik (SVM, RF, KNN, LR, MLP) menunjukkan akurasi tinggi pada test set.
* **Weighted Voting Ensemble** menjadi baseline ensemble terbaik di antara model klasik (akurasi & F1-weighted ≈ **90%+**).
* Insight utama: kombinasi fitur warna (HSV) dan tekstur (GLCM + LBP) cukup efektif untuk membedakan tiga kelas daun kopi.

---

## 3. Pipeline Deep Learning (CNN)

### 3.1. Data Loading

Menggunakan `ImageDataGenerator`:

* **Train**:

  * `rescale=1./255`
  * `rotation_range=20`
  * `width_shift_range=0.2`
  * `height_shift_range=0.2`
  * `shear_range=0.2`
  * `zoom_range=0.2`
  * `horizontal_flip=True`

* **Val/Test**:

  * Hanya `rescale=1./255` (tanpa augmentasi).

Input size: **224×224×3**, `batch_size = 32`.

### 3.2. Arsitektur CNN

Model `Sequential` murni (tanpa transfer learning):

```text
Conv2D(32, 3×3) + MaxPool
Conv2D(64, 3×3) + MaxPool
Conv2D(128, 3×3) + MaxPool
Conv2D(256, 3×3) + MaxPool
Flatten
Dense(512, relu) + Dropout(0.5)
Dense(256, relu) + Dropout(0.3)
Dense(num_classes, softmax)
```

* Optimizer  : `adam`
* Loss       : `categorical_crossentropy`
* Metrics    : `accuracy`

### 3.3. Training Setup

* **Epoch max**: 50
* **Callbacks**:

  * `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`
  * `ModelCheckpoint('best_coffee_leaf_cnn_model.h5', save_best_only=True)`

### 3.4. Hasil CNN

Evaluasi pada **450 citra uji**:

* `Test Loss`   ≈ **0.1415**
* `Test Accuracy` ≈ **0.9511** (≈ **95,1%**)

Dari classification report (CNN):

* `Daun_Bercak` : precision 0.97, recall 0.92, F1 0.94
* `Daun_Karat`  : precision 0.93, recall 0.95, F1 0.94
* `Daun_Sehat`  : precision 0.96, recall 0.98, F1 0.97
* **Macro / weighted F1 ≈ 0.95**

Artinya:

* Model CNN berhasil mengklasifikasikan **428 dari 450** citra uji dengan benar.
* Performa antar kelas relatif seimbang (≈92–98%), sehingga CNN cukup stabil untuk ketiga kelas daun kopi.

---

## 4. Cara Menjalankan

Proyek ini dikembangkan di **Google Colab** dengan dataset di Google Drive, tetapi dapat dijalankan lokal dengan sedikit penyesuaian path.

### 4.1. Kebutuhan Paket

Buat file `requirements.txt` (opsional) kira-kira berisi:

```txt
tensorflow
numpy
pandas
scikit-learn
opencv-python
matplotlib
seaborn
scikit-image
split-folders
transformers
datasets
timm
torch
peft
groq
```

Lalu install:

```bash
pip install -r requirements.txt
```

### 4.2. ML Klasik

1. Pastikan dataset sudah di-split ke `dataset_coffee_split/` seperti struktur di atas.

2. Jalankan notebook / script ML klasik secara berurutan:

   * Split dataset (jika belum).
   * Load dataset & EDA.
   * Preprocessing gambar (normalisasi + augmentasi).
   * Ekstraksi fitur (HSV, GLCM, LBP).
   * Preprocessing fitur (scaling, optional PCA).
   * Seleksi fitur (`SelectKBest`).
   * Training multi-model + GridSearchCV.
   * Evaluasi + Error Analysis + Ensemble.

3. Model ensemble terbaik dapat disimpan, misalnya:

   ```python
   import joblib
   joblib.dump(weighted_ensemble, 'best_coffee_classifier.pkl')
   joblib.dump(le, 'label_encoder.pkl')
   joblib.dump(final_scaler, 'feature_scaler.pkl')
   ```

### 4.3. Deep Learning (CNN)

1. Pastikan `DATASET_PATH` di-set ke folder split (`dataset_coffee_split`).

2. Jalankan bagian:

   * Load dataset dengan `ImageDataGenerator`.
   * EDA distribusi kelas & contoh gambar.
   * Definisi arsitektur CNN.
   * Training (`model.fit(...)`).
   * Evaluasi pada test set + confusion matrix.

3. Model terbaik otomatis disimpan sebagai:

   ```text
   best_coffee_leaf_cnn_model.h5
   ```

### 4.4. Visual Transformer (ViT), LoRA, dan XAI

1. Pastikan struktur folder dataset sama (`dataset_coffee_split`).
2. Jalankan notebook **ViT / XAI + LoRA**, yang mencakup:

   * EDA dataset (distribusi kelas, contoh gambar, histogram warna, cek blur).
   * Preprocessing & augmentasi untuk train set (flip, rotation, color jitter).
   * Fine-tuning ViT dengan `Trainer` (HuggingFace).
   * Fine-tuning lanjutan menggunakan **LoRA (PEFT)** pada checkpoint ViT yang sudah dilatih.
   * Evaluasi model ViT dan ViT+LoRA (accuracy, classification report, confusion matrix).
   * Penerapan **Grad-CAM** pada model ViT/LoRA untuk highlight area penting pada daun.
   * Integrasi dengan **Groq API (LLM)** untuk memberikan penjelasan penyakit dalam bentuk teks.

---

## 5. Struktur (Contoh)

Silakan sesuaikan dengan struktur repo kamu, tetapi secara konsep bisa seperti:

```text
.
├── README.md
├── classic_ml/
│   └── coffee_leaf_ml_classic.ipynb
├── deep_learning/
│   └── coffee_leaf_cnn.ipynb
├── vit_lora_xai/
│   └── XAI_LoRA_ViT_daun_kopi.ipynb
└── dataset/  (opsional, biasanya di Google Drive)
```

---

## 6. Visual Transformer (ViT) & LoRA

### 6.1. Overview

Pada tahap ini digunakan **Vision Transformer (ViT)** sebagai model berbasis **Transformers untuk citra**, dengan alur:

* Load model pre-trained `google/vit-base-patch16-224-in21k`.
* Fine-tuning pada dataset daun kopi (3 kelas).
* Menyimpan checkpoint terbaik berdasarkan akurasi.
* Melanjutkan fine-tuning dengan **LoRA** untuk efisiensi parameter.

### 6.2. Preprocessing & Augmentasi untuk ViT

* Seluruh gambar dikonversi ke **RGB** dan di-*resize* otomatis oleh `ViTImageProcessor`.
* Distribusi kelas divisualisasikan (bar chart + pie chart).
* Contoh gambar dari tiap kelas ditampilkan.
* Cek kualitas gambar:

  * Histogram warna (RGB) per kelas.
  * Deteksi gambar buram menggunakan standar deviasi intensitas grayscale.
* Augmentasi training menggunakan:

  ```python
  train_augment = T.Compose([
      T.RandomHorizontalFlip(p=0.5),
      T.RandomVerticalFlip(p=0.5),
      T.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.4),
      T.RandomRotation(30),
  ])
  ```

### 6.3. Training ViT

* Model: `ViTForImageClassification`.

* Optimizer & scheduler dikelola otomatis oleh `Trainer`.

* Hyperparameter kunci:

  * `batch_size = 16`
  * `num_train_epochs = 5`
  * `learning_rate = 3e-5`
  * `metric_for_best_model = 'accuracy'`
  * `load_best_model_at_end = True`

* Evaluasi:

  * Akurasi test set
  * `classification_report` (precision, recall, F1-score per kelas)
  * Confusion matrix (heatmap)

### 6.4. Fine-Tuning dengan LoRA

* Checkpoint terbaik ViT (hasil training full) diekstrak dari zip (`vit-daunkopi-checkpoints.zip`).

* LoRA dikonfigurasikan dengan:

  ```python
  lora_config = LoraConfig(
      task_type=TaskType.IMAGE_CLASSIFICATION,
      r=8,
      lora_alpha=16,
      lora_dropout=0.1,
      target_modules=["query", "key", "value"]
  )
  ```

* Model ViT dibungkus menjadi **PeftModel** (ViT + LoRA).

* Hanya subset kecil parameter yang dilatih, sehingga:

  * Training lebih cepat.
  * Konsumsi memori lebih ringan.
  * Performa tetap sangat tinggi.

* Training LoRA:

  * `num_train_epochs = 3`
  * `learning_rate = 1e-4`
  * Metode evaluasi sama (accuracy di validation dan test).

### 6.5. Hasil ViT + LoRA

* ViT full fine-tune memberikan akurasi test tinggi (≈95%+).
* ViT + LoRA menghasilkan:

  * Akurasi test set ≈ **96–97%**.
  * Precision, recall, dan F1-score seimbang antar kelas.
* LoRA terbukti:

  * Menghemat parameter yang dilatih.
  * Meningkatkan atau mempertahankan performa model.
  * Cocok untuk scenario fine-tuning lanjutan atau deployment resource terbatas.

---

## 7. Explainable AI (XAI): Grad-CAM & Groq LLM

### 7.1. Grad-CAM untuk ViT / LoRA

Untuk menjelaskan **mengapa** model memprediksi suatu kelas (bukan hanya **apa** hasil prediksinya), digunakan **Grad-CAM** yang dimodifikasi untuk arsitektur ViT:

* Hook dipasang pada layer encoder terakhir ViT (`vit.encoder.layer[-1].output`).

* Saat forward-backward:

  * Disimpan **activations** dan **gradients**.
  * Dibuat peta pentingnya (Class Activation Map) berdasarkan rata-rata gradien dan aktivasi.
  * Token CLS diabaikan, hanya patch token yang dipakai.
  * Peta 1D di-*reshape* ke bentuk 2D (grid patch), lalu di-*resize* ke ukuran gambar asli.

* Hasil Grad-CAM di-*overlay* di atas gambar daun:

  * Daerah yang paling berkontribusi pada prediksi akan tampak berwarna lebih “panas” (merah/kuning).
  * Sangat membantu untuk melihat apakah model fokus ke area bercak, karat, atau bagian lain.

### 7.2. Penjelasan Berbasis Aturan (Rule-Based XAI)

Di dalam kelas `VitGradCAM` tersedia juga fungsi:

* `generate_xai_explanation(label_idx)` yang mengembalikan penjelasan singkat per kelas:

  * `Daun_Bercak` → bercak coklat akibat jamur.
  * `Daun_Karat` → karat daun oleh *Hemileia vastatrix*.
  * `Daun_Sehat` → daun hijau tanpa gejala penyakit.

Penjelasan ini dapat digunakan sebagai deskripsi singkat untuk pengguna akhir (misalnya petani atau praktisi kebun).

### 7.3. Integrasi Groq LLM untuk Penjelasan Mendalam

Untuk penjelasan yang lebih kaya dan naratif, digunakan **Groq API** dengan model **`llama-3.3-70b-versatile`**:

* Fungsi `get_additional_explanation(disease_name)`:

  * Mengirim prompt:
    `"Jelaskan secara mendalam mengenai penyakit {disease_name} pada tanaman kopi dan bagaimana cara menanganinya."`
  * Model kemudian mengembalikan penjelasan panjang terkait:

    * Penyebab penyakit.
    * Gejala visual pada daun.
    * Dampak ke tanaman.
    * Rekomendasi penanganan dan pencegahan.

* Fungsi `generate_xai_explanation(predicted_label)`:

  * Menggabungkan:

    * Penjelasan singkat (rule-based).
    * Penjelasan panjang hasil Groq (LLM).

### 7.4. Pipeline XAI Lengkap

Fungsi `explain_with_gradcam_and_xai(image, model_lora, image_processor)` melakukan:

1. Prediksi kelas gambar daun kopi dengan model ViT + LoRA.

2. Menghasilkan **Grad-CAM** dan menampilkan overlay pada gambar.

3. Mengambil:

   * Nama penyakit.
   * Penjelasan singkat (rule-based).
   * Penjelasan tambahan dari Groq (LLM).

4. Menampilkan:

   * Gambar asli.
   * Peta Grad-CAM (heatmap).
   * Overlay Grad-CAM + gambar.
   * Teks penjelasan penyakit + cara penanganan.

Dengan demikian, sistem tidak hanya melakukan klasifikasi, tetapi juga:

* Menjelaskan **area mana di daun** yang menjadi dasar keputusan model.
* Memberikan **penjelasan teks yang mudah dipahami** tentang penyakit dan langkah penanganannya.

---

Repo ini dengan demikian mencakup:

* **ML Klasik** → fitur rekayasa + ensemble.
* **CNN** → end-to-end deep learning dari citra ke label.
* **ViT + LoRA** → model transformer modern yang efisien.
* **XAI (Grad-CAM + Groq)** → transparansi dan interpretabilitas untuk pengguna akhir.
