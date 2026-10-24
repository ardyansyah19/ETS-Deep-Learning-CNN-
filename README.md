# ETS Deep Learning — MLP untuk Pengenalan Angka Tulisan Tangan MNIST

## Isi paket
| Berkas | Fungsi |
|---|---|
| `ETS_DeepLearning_MNIST_MLP.ipynb` | Notebook utama (TensorFlow/Keras) |
| `artifacts/` | Model terlatih, riwayat pelatihan kedua konfigurasi, grafik, konfigurasi (terisi setelah notebook dijalankan) |
| `requirements.txt` | Pustaka yang dibutuhkan |
| `README.md` | Petunjuk menjalankan |

## A. Menjalankan di Google Colab (disarankan)
1. Buka https://colab.research.google.com → **File → Upload notebook** → pilih `ETS_DeepLearning_MNIST_MLP.ipynb`.
2. (Opsional) **Runtime → Change runtime type → GPU**. MLP ini juga cepat di CPU.
3. Pastikan `RETRAIN = True` pada Bagian 1, lalu **Runtime → Run all**.
   Data dimuat otomatis via `keras.datasets.mnist` (tidak perlu mengunduh CSV).
4. Setelah selesai, folder `artifacts/` (di panel kiri Colab) berisi model dan riwayat pelatihan.
   Sel terakhir membuat `hasil_pelatihan_artifacts.zip` dan mengunduhnya otomatis.
5. Satukan: notebook + isi `artifacts/` + `README.md` + `requirements.txt` dalam **satu ZIP** untuk dikumpulkan.

> Runtime Colab dihapus saat sesi berakhir. **Segera unduh `artifacts/`** setelah pelatihan.

## B. Memakai CSV (opsional)
Ubah `"data_source": "csv"` di `CONFIG`, taruh `mnist_train.csv` dan `mnist_test.csv` di folder `data/`.
Notebook mendeteksi header otomatis dan memeriksa bentuk data (60000×785 dan 10000×785).
Pilih **satu** sumber saja (Keras **atau** CSV).

## C. Saat demo ETS (tanpa melatih ulang)
1. Unggah ke Colab: notebook + folder `artifacts/` (berisi `*.keras` dan `*.json`).
2. Ubah `RETRAIN = False` di Bagian 1, lalu **Run all**.
3. Notebook memuat model dan riwayat dari `artifacts/`, menampilkan ulang grafik, pemilihan model,
   akurasi pengujian, dan fitur prediksi.

## D. Pengaturan (juga tersimpan di `artifacts/config.json`)
Seed 42 · split berstrata 80/20 · 784 → 256 → 128 → 10 · ReLU & Softmax · loss `sparse_categorical_crossentropy` ·
Adam, lr = 0.001 · batch 128 · maks 30 epoch · Dropout 0.3 (model regularisasi) ·
Early Stopping: `val_loss`, patience 5, `restore_best_weights=True`.
Kedua model memakai bobot awal yang identik (dibuktikan dengan `assert` di notebook).

## E. Pengaturan (juga tersimpan di `artifacts/config.json`)
Seed 42 · split berstrata 80/20 · 784 → 256 → 128 → 10 · ReLU & Softmax · loss `sparse_categorical_crossentropy` ·
Adam, lr = 0.001 · batch 128 · maks 30 epoch · Dropout 0.3 (model regularisasi) ·
Early Stopping: `val_loss`, patience 5, `restore_best_weights=True`.
Kedua model memakai bobot awal yang identik (dibuktikan dengan `assert` di notebook).

## E. Catatan
- Data uji hanya dipakai untuk evaluasi akhir model terpilih dan fitur prediksi.
- Hasil angka dapat sedikit berbeda antar-perangkat/versi TensorFlow; yang dinilai adalah konsistensi dengan `artifacts/` Anda sendiri.
