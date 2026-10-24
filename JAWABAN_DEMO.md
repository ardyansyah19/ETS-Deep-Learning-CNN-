# Panduan Demo dan Jawaban Lisan ETS Deep Learning

Dokumen ini untuk belajar. Pahami alasannya, jangan dihafal kata per kata. Angka di bawah adalah contoh dari satu kali uji; **ganti dengan angka dari hasil Anda sendiri**.

---

## 1. Urutan demo (sesuai soal bagian B)
| # | Yang diminta | Yang Anda lakukan di notebook |
|---|---|---|
| 1 | Jalankan program, tunjukkan contoh data + label | Bagian 2–4: grid 10 digit dengan label |
| 2 | Arsitektur dan fungsi tiap lapisan | Bagian 5: `model.summary()` |
| 3 | Hasil pelatihan dua model + pengaruh regularisasi | Bagian 6–7: grafik dan tabel ringkasan |
| 4 | Prediksi minimal 5 gambar pilihan dosen | Bagian 10: `predict_many([idx1, ..., idx5])` atau widget |
| 5 | Jelaskan kode: normalisasi, aktivasi, loss, optimizer, prediksi | Lihat bagian 4 di bawah |
| 6 | Tanya jawab | Bagian 5 di bawah |

Saat demo: `RETRAIN = False` → *Run all* (memuat model dari `artifacts/`, tidak melatih ulang).

---

## 2. Arsitektur dan fungsi tiap lapisan
`784 → Dense(256, ReLU) → [Dropout] → Dense(128, ReLU) → [Dropout] → Dense(10, Softmax)`

| Lapisan | Fungsi |
|---|---|
| Input 784 | Gambar 28×28 diratakan menjadi 784 fitur (satu fitur = satu piksel) |
| Hidden 1 (256, ReLU) | Mempelajari kombinasi piksel menjadi pola sederhana (garis, lengkung) |
| Hidden 2 (128, ReLU) | Menggabungkan pola sederhana menjadi pola yang lebih abstrak (bagian angka) |
| Dropout (hanya model 2) | Saat pelatihan, 30% neuron dimatikan acak di tiap langkah → mengurangi overfitting |
| Output 10 (Softmax) | 10 probabilitas (angka 0–9), jumlahnya 1; prediksi = probabilitas terbesar |

**Jumlah parameter:** (784×256+256) + (256×128+128) + (128×10+10) = 200.960 + 32.896 + 1.290 = **235.146**. Dropout tidak punya parameter.

---

## 3. Konsep inti (jawaban singkat)
- **Mengapa dibagi 255?** Piksel 0–255 diubah ke 0–1 agar skala fitur seragam, gradien stabil, pelatihan lebih cepat dan konvergen.
- **Mengapa ReLU?** `max(0, x)`: sederhana, murah dihitung, mengurangi masalah vanishing gradient dibanding sigmoid/tanh, dan memberi non-linearitas. Tanpa non-linearitas, banyak lapisan setara satu lapisan linear.
- **Mengapa Softmax di output?** Mengubah 10 skor menjadi distribusi probabilitas: `p_i = e^{z_i} / Σ e^{z_j}`.
- **Mengapa `sparse_categorical_crossentropy`?** Label berupa bilangan bulat (0–9). Jika label di-one-hot, pakai `categorical_crossentropy`. Hasil matematisnya sama: `L = -log(p_kelas_benar)`. Semakin yakin pada kelas benar, loss mendekati 0.
- **Adam?** Optimizer yang memadukan momentum dan learning rate adaptif per parameter. lr = 0,001 (nilai umum yang stabil).
- **Epoch, batch, iterasi?** Epoch = satu putaran seluruh data latih. Batch 128. Data latih 48.000 → 48.000/128 = **375 iterasi per epoch**.
- **Mengapa ada data validasi?** Untuk memantau generalisasi, menjalankan Early Stopping, dan memilih model **tanpa menyentuh data uji**.
- **Mengapa split berstrata?** Proporsi tiap digit di data latih dan validasi sama, sehingga validasi representatif.
- **Mengapa bobot awal dibuat sama?** Agar perbedaan hasil benar-benar berasal dari Dropout + Early Stopping, bukan dari inisialisasi acak (perbandingan adil).
- **Early Stopping?** Memantau `val_loss`; jika tidak membaik selama 5 epoch (`patience=5`), pelatihan berhenti. `restore_best_weights=True` mengembalikan bobot epoch terbaik, bukan epoch terakhir.
- **Dropout saat prediksi?** Nonaktif. Keras memakai *inverted dropout* (nilai diskalakan saat pelatihan), sehingga saat inferensi semua neuron dipakai.

---

## 4. Penjelasan kode (untuk poin B.5)
| Topik | Lokasi | Penjelasan |
|---|---|---|
| Normalisasi | fungsi `normalize()` (Bagian 3) | `a.astype("float32") / 255.0` → rentang 0–1 |
| Flatten | Bagian 2 | `reshape(len(x), -1)` → 28×28 menjadi 784 |
| Split berstrata | `train_test_split(..., stratify=y_full, random_state=SEED)` | 80/20, proporsi kelas terjaga |
| Aktivasi | `build_mlp()` | `activation="relu"` di hidden, `"softmax"` di output |
| Loss | `model.compile(loss=...)` | `sparse_categorical_crossentropy` (label integer) |
| Optimizer | `keras.optimizers.Adam(learning_rate=...)` | Memperbarui bobot berdasarkan gradien loss |
| Early Stopping | `keras.callbacks.EarlyStopping(...)` | `monitor="val_loss"`, `restore_best_weights=True` |
| Prediksi | `predict_test_image()` | `model.predict` → 10 probabilitas; `np.argmax` → angka prediksi |
| Pemilihan model | Bagian 8 | `min(...)` atas `val_loss` validasi; data uji tidak dipakai |

---

## 5. Membaca grafik (Hasil Aktual Pelatihan)
| Metrik | Model Dasar (Tanpa Dropout & ES) | Model Regularisasi (Dropout 0.3 + ES) |
|---|---|---|
| **Jumlah Epoch Berjalan** | 30 epoch (maksimal) | **11 epoch** (berhenti otomatis via Early Stopping) |
| **Epoch Bobot Terbaik** | Epoch 6 (val_loss 0,0906) | **Epoch 6 (val_loss 0,0803)** |
| **Akurasi Latih** | 99,69% | 98,89% |
| **Akurasi Validasi** | 97,63% | **97,78%** |
| **Selisih (Latih - Val)** | 2,05% (mulai melebar) | **1,11%** (selisih sangat rapat/ketat) |
| **Loss Latih** | 0,0106 | 0,0367 |
| **Loss Validasi** | 0,1501 (memburuk akibat overfitting) | **0,0803 (hampir separuh dari model dasar!)** |
| **Akurasi Data Uji (Test)** | - | **97,65% (Loss Uji: 0,0757)** |

**Penjelasan Grafik Saat Ditanya Dosen:**
- **Model dasar:** Akurasi latih melesat hingga 99,69% dan loss latih terus ditekan mendekati nol (0,0106). Namun, perhatikan kurva *loss validasi*: titik terendahnya ada di **epoch 6 (0,0906)**, setelah itu justru **berbalik naik terus menerus hingga 0,1501 di epoch 30**. Selisih performa latih vs validasi melebar. Ini adalah bukti nyata terjadinya **overfitting** (jaringan mulai menghafal noise data latih).
- **Model Dropout + Early Stopping:** 
  1. *Dropout (rate 0,3)* memaksa neuron tidak saling bergantung (*co-adaptation*), sehingga kurva validasi jauh lebih stabil dan generalisasinya lebih kuat (selisih akurasi hanya 1,11%).
  2. *Early Stopping* memantau `val_loss` dengan `patience=5`. Karena setelah epoch 6 tidak ada perbaikan loss validasi selama 5 epoch berturut-turut, pelatihan dihentikan otomatis pada **epoch 11**.
  3. Berkat `restore_best_weights=True`, bobot yang disimpan bukan bobot epoch 11, melainkan bobot terbaik dari **epoch 6** yang memiliki `val_loss = 0,0803`.
- **Hasil Akhir:** Model Regularisasi terpilih sebagai model terbaik berdasarkan kriteria validasi, dan saat diuji pada 10.000 data pengujian yang belum pernah dilihat sama sekali, model mencapai akurasi **97,65%** dengan loss **0,0757**.

---

## 6. Pertanyaan yang mungkin muncul
1. **Mengapa data uji tidak boleh dipakai memilih model?** Jika dipakai, data uji ikut "bocor" ke keputusan kita, sehingga akurasinya menjadi terlalu optimis dan tidak lagi mewakili data baru.
2. **Mengapa Softmax, bukan Sigmoid?** Ini klasifikasi satu kelas dari 10 (saling eksklusif); Softmax menghasilkan distribusi yang jumlahnya 1.
3. **Apa yang terjadi bila learning rate terlalu besar/kecil?** Terlalu besar: loss berosilasi atau divergen. Terlalu kecil: belajar sangat lambat.
4. **Apa itu overfitting dan bagaimana mendeteksinya?** Performa latih jauh lebih baik dari validasi; terlihat dari celah antar kurva dan val_loss yang naik.
5. **Cara lain mengurangi overfitting?** Menambah data/augmentasi, mengecilkan model, L2 regularization, Batch Normalization.
6. **Mengapa MLP kurang ideal untuk gambar?** Meratakan gambar membuang struktur spasial; CNN memanfaatkan kedekatan piksel dan berbagi bobot, sehingga lebih efisien dan akurat.
7. **Mengapa `shuffle=True`?** Agar urutan data tiap epoch acak, sehingga pelatihan lebih stabil. Seed dijaga agar hasil dapat diulang.
8. **Kenapa batch size 128?** Kompromi antara kestabilan gradien dan kecepatan; nilai yang umum dipakai.
9. **Apa arti probabilitas 99,8% pada prediksi?** Keyakinan model; belum tentu benar. Model bisa yakin tapi salah.
10. **Bagaimana model disimpan dan dimuat ulang?** `model.save("x.keras")` dan `keras.models.load_model("x.keras")`; riwayat disimpan dalam JSON.

---

## 7. Tips praktis hari-H
- Jalankan seluruh notebook **sekali sebelum berangkat** dan pastikan `artifacts/` lengkap, termasuk di laptop/Drive Anda.
- Siapkan 5–6 indeks gambar uji, termasuk satu yang diprediksi salah (cari lewat confusion matrix) untuk menunjukkan pemahaman.
- Jika dosen meminta indeks tertentu: `predict_test_image(1234)`.
- Jika internet bermasalah, gunakan mode CSV atau salinan lokal dan pastikan `RETRAIN = False`.
