# Hasil Benchmark: UCS vs A*

Tabel komparasi metrik kinerja algoritma UCS dan A* pada jaringan jalan perkotaan (CERTAN-01 Milestone 1).


| Skenario | Algoritma | Biaya Optimal (Rp) | Jarak (km) | Simpul Dijelajahi | Runtime (ms) |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Skenario 1: Depo Utara → Perumahan Griya | UCS | Rp 19,400 | 11.50 | 10 | 0.1058 |
|  | A\* | Rp 19,400 | 11.50 | 7 | 0.0929 |
| Skenario 2: Depo Utara → Taman Kota Selatan | UCS | Rp 27,800 | 16.50 | 12 | 0.0526 |
|  | A\* | Rp 27,800 | 16.50 | 11 | 0.0668 |
| Skenario 3: Depo Utara → Jl. Ahmad Yani Selatan | UCS | Rp 18,300 | 12.00 | 9 | 0.0376 |
|  | A\* | Rp 18,300 | 12.00 | 8 | 0.0427 |
| Skenario 4: Jl. Veteran → Perumahan Griya | UCS | Rp 14,900 | 8.50 | 10 | 0.0421 |
|  | A\* | Rp 14,900 | 8.50 | 5 | 0.0335 |

> **Keterangan**: *Simpul Dijelajahi* = jumlah simpul yang dieksplorasi (*nodes expanded*) hingga solusi ditemukan.