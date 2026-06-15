# Konsep-Dasar-Kecerdasan-Artifisial
Analisis Klasifikasi Sampah Menggunakan Decision Tree dan Random Forest

# Klasifikasi Sampah (Organik & Anorganik) Menggunakan Decision Tree dan Random Forest 

Proyek ini merupakan implementasi untuk mengklasifikasikan jenis sampah menjadi dua kategori utama: **Organik** dan **Anorganik** melalui pemrosesan citra digital. Algoritma **Decision Tree** dan **Random Forest** dalam proyek ini dibangun dari awal (*from scratch*) menggunakan Python dan NumPy, tanpa bergantung pada *library* bawaan.

## Fitur Utama
* **Algoritma Custom**: Implementasi logika node, perhitungan *Entropy*, *Information Gain*, dan pembuatan pohon (Tree Building) secara mandiri.
* **Ekstraksi Fitur Citra**: Memanfaatkan OpenCV untuk mengekstrak nilai rata-rata RGB, tingkat kecerahan (*brightness*), serta komponen HSV.
* **Komparasi Model**: Membandingkan tingkat akurasi dan waktu eksekusi antara *Decision Tree* tunggal dan *Random Forest* (kumpulan pohon/ensemble).
* **Prediksi Custom**: Mendukung pengujian pada gambar spesifik yang berada di luar dataset (misal: `pisang.jpeg`).

## Struktur Direktori & Dataset
Agar program dapat berjalan dengan baik, Anda harus menyiapkan dataset gambar di dalam folder `dataset_sampah` dengan struktur sub-direktori berdasarkan label (kategori) sebagai berikut:

```text
direktori_proyek/
│
├── 1.py                # Script ekstraksi fitur dasar (RGB + Brightness)
├── 2.py                # Script ekstraksi fitur kompleks (+ HSV) & komputasi waktu
├── pisang.jpeg         # Contoh gambar untuk tes prediksi eksternal
│
└── dataset_sampah/     # Direktori dataset utama
    ├── anorganik/      # Berisi gambar-gambar sampah plastik, kaleng, botol, dll.
    │   ├── botol_1.jpg
    │   ├── plastik_2.png
    │   └── ...
    │
    └── organik/        # Berisi gambar-gambar sisa makanan, daun, buah, dll.
        ├── apel_1.jpg
        ├── daun_2.jpeg
        └── ...
```

## Perbedaan File 1.py dan 2.py

| Fitur / Parameter | 1.py | 2.py |
|-----------------|-----|-----|
| Ekstraksi Fitur | 4 Fitur (R, G, B, Brightness) | 6 Fitur (R, G, B, Brightness, Hue Mean, Hue Std) | 
| Max Depth (Tree) | 5 | 7 |
| N-Estimators (RF) | 20 | 23 |

## Prasyarat Instalasi
Pastikan sudah terpasang library berikut:
```bash
pip install opencv-python numpy scikit-learn
```
*`scikit-learn` hanya digunakan untuk modul `train_test_split` demi membagi data latih dan data uji secara proporsional).*

## Cara Penggunaan
* Pastikan folder `dataset_sampah` beserta sub-foldernya sudah terisi dengan gambar.
* Buka terminal atau *command prompt*, arahkan ke direktori proyek, lalu jalankan salah satu file
  ```bash
  python 1.py
  # atau
  python 2.py
  ```
* Untuk mengetes gambar lain di luar dataset, ubah argumen pada fungsi `tes_prediksi()` di baris paling bawah kode program:
  ```bash
  tes_prediksi('nama_file_gambar.jpg')
  tes_prediksi2('nama_file_gambar.jpg')
  ```
