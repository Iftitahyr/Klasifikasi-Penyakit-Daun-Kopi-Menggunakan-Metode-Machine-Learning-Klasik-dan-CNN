# Klasifikasi Penyakit Daun Kopi Menggunakan Metode Machine Learning Klasik, Deep Learning dan Visual Transformer 
 
# Coffee Leaf Disease Classification

Klasifikasi Penyakit Daun Kopi dengan **Machine Learning Klasik & Deep Learning (CNN)**

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
└── dataset/  (opsional, biasanya di Google Drive)
```

Berikut adalah format **README.md** yang telah disesuaikan dengan proyek klasifikasi penyakit daun kopi menggunakan **Visual Transformer** (ViT):

---

# Klasifikasi Penyakit Daun Kopi Menggunakan Visual Transformer (pre-trained)

## Coffee Leaf Disease Classification

Proyek ini membangun sistem klasifikasi citra daun kopi menjadi **3 kelas**:

* `Daun_Sehat` (Healthy)
* `Daun_Bercak` (Cercospora coffeicola)
* `Daun_Karat` (Hemileia vastatrix)

Pendekatan yang digunakan:

1. **Visual Transformer (ViT) Pre-trained**

   * Menggunakan model pre-trained **Visual Transformer** dari Google.
   * Fine-tuning model untuk tugas spesifik klasifikasi penyakit daun kopi.

2. **Data Preprocessing dan Augmentasi**

   * Memproses citra untuk memastikan kualitas data.
   * Augmentasi data pada dataset pelatihan untuk meningkatkan kemampuan generalisasi model.

---

## 1. Dataset

* Dataset terdiri dari citra daun kopi dengan 3 kelas:

  * **Daun_Sehat**
  * **Daun_Bercak**
  * **Daun_Karat**

* Dataset disimpan di Google Drive dan dibagi menjadi:

  * **Train**: 80%
  * **Validation**: 10%
  * **Test**: 10%

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

Script untuk melakukan pembagian dataset (splitting) menggunakan library `splitfolders`:

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

## 2. Data Preprocessing

### 2.1. Visualisasi dan Preprocessing Gambar

Sebelum pelatihan, data citra diproses dengan langkah-langkah berikut:

* **Visualisasi**: Menampilkan contoh gambar dari setiap kelas untuk memastikan distribusi yang seimbang.
* **Pembersihan Data**: Menghapus gambar yang buram dengan menggunakan modul PIL untuk mendeteksi ketajaman gambar.
* **Resize dan Normalisasi**: Mengubah ukuran gambar menjadi **224x224** piksel dan menormalkan nilai piksel ke rentang **[0, 1]**.

```python
sizes, modes = [], []
for cls in classes:
    class_folder = os.path.join(data_root, 'train', cls)
    for f in os.listdir(class_folder)[:15]:
        img_path = os.path.join(class_folder, f)
        with Image.open(img_path) as img:
            sizes.append(img.size)
            modes.append(img.mode)
sizes = np.array(sizes)
```

### 2.2. Augmentasi Data

Augmentasi dilakukan hanya pada dataset **training** untuk meningkatkan keberagaman data dan mengurangi overfitting, dengan menggunakan teknik berikut:

* **Flip Horizontal dan Vertical**
* **Random Rotation** hingga 30 derajat
* **Color Jitter** (brightness, contrast, saturation)
* **Zoom** ringan

```python
train_augment = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.4),
    T.RandomRotation(30),
])
```

---

## 3. Model Training dengan Visual Transformer (ViT)

### 3.1. Memuat Model Pre-trained

Untuk tugas ini, kami menggunakan model **Visual Transformer (ViT)** yang telah dilatih sebelumnya pada dataset **ImageNet** dan kemudian di-fine-tune untuk klasifikasi tiga kelas daun kopi.

```python
model_name = 'google/vit-base-patch16-224-in21k'
image_processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={str(i): lbl for i, lbl in enumerate(labels)},
    label2id={lbl: str(i) for i, lbl in enumerate(labels)}
)
```

### 3.2. Pelatihan Model

Model dilatih dengan **batch size 16** dan **learning rate 3e-5** selama 5 epoch. Pelatihan menggunakan **Trainer** dari **HuggingFace** dengan pengukuran metrik **accuracy**.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)
trainer.train()
```

### 3.3. Evaluasi Model

Evaluasi dilakukan pada **dataset test** untuk memperoleh hasil dari **classification report** dan **confusion matrix**:

```python
cm = confusion_matrix(true_labels, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Prediksi Model')
plt.ylabel('Label Asli (Aktual)')
plt.title('Confusion Matrix - Klasifikasi Penyakit Daun Kopi')
plt.show()
```

---

## 4. Hasil Model

Evaluasi pada **450 citra uji** menghasilkan **Test Accuracy ≈ 95%**. Berikut adalah **classification report** yang menunjukkan kinerja model pada masing-masing kelas:

* `Daun_Bercak`: Precision 0.97, Recall 0.92, F1-Score 0.94
* `Daun_Karat`: Precision 0.93, Recall 0.95, F1-Score 0.94
* `Daun_Sehat`: Precision 0.96, Recall 0.98, F1-Score 0.97

Model ini menunjukkan stabilitas yang baik di seluruh kelas daun kopi.

---

## 5. Cara Menjalankan

Proyek ini dikembangkan di **Google Colab**, namun Anda dapat menjalankannya di lingkungan lokal dengan sedikit penyesuaian pada path.

### 5.1. Kebutuhan Paket

Buat file `requirements.txt` yang berisi daftar paket yang dibutuhkan, seperti:

```txt
transformers
datasets
timm
scikit-learn
torch
pandas
matplotlib
seaborn
opencv-python
```

Lalu install paket-paket yang dibutuhkan dengan perintah:

```bash
pip install -r requirements.txt
```

### 5.2. Langkah-langkah untuk Menjalankan

1. **Menyiapkan Dataset**: Pastikan dataset sudah di-split ke dalam folder dengan struktur yang benar.
2. **Pelatihan Model**: Jalankan script untuk melatih model menggunakan **Visual Transformer**.
3. **Evaluasi**: Evaluasi model pada test set dan tampilkan hasil confusion matrix serta classification report.
4. **Prediksi Gambar**: Coba prediksi gambar daun kopi dengan mengunggah citra menggunakan antarmuka yang disediakan.

---
