import numpy as np

# Fungsi aktivasi sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Turunan fungsi aktivasi sigmoid
def turunan_sigmoid(x):
    return x * (1 - x)

# Dataset input
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# Dataset output (target)
y = np.array([[0],
              [1],
              [1],
              [0]])

# Inisialisasi bobot secara acak
np.random.seed(1)  # Untuk hasil acak yang tetap sama setiap kali dijalankan
bobot0 = 2 * np.random.random((3, 4)) - 1  # Bobot antara lapisan input dan tersembunyi
bobot1 = 2 * np.random.random((4, 1)) - 1  # Bobot antara lapisan tersembunyi dan output

# Pelatihan neural network
for iterasi in range(60000):  # Ulang sebanyak 60.000 kali

    # **Langkah Forward Propagation**
    lapisan0 = X  # Lapisan input
    lapisan1 = sigmoid(np.dot(lapisan0, bobot0))  # Lapisan tersembunyi
    lapisan2 = sigmoid(np.dot(lapisan1, bobot1))  # Lapisan output

    # Hitung error (selisih antara prediksi dan target)
    error_lapisan2 = y - lapisan2

    # Cetak error setiap 10.000 iterasi untuk memantau progres
    if (iterasi % 10000) == 0:
        print(f"Error setelah {iterasi} iterasi: {np.mean(np.abs(error_lapisan2))}")

    # **Langkah Backpropagation**
    delta_lapisan2 = error_lapisan2 * turunan_sigmoid(lapisan2)  # Gradien lapisan output
    error_lapisan1 = delta_lapisan2.dot(bobot1.T)  # Error di lapisan tersembunyi
    delta_lapisan1 = error_lapisan1 * turunan_sigmoid(lapisan1)  # Gradien lapisan tersembunyi

    # Update bobot
    bobot1 += lapisan1.T.dot(delta_lapisan2)  # Update bobot dari lapisan tersembunyi ke output
    bobot0 += lapisan0.T.dot(delta_lapisan1)  # Update bobot dari input ke lapisan tersembunyi

# Hasil setelah pelatihan selesai
print("\nOutput setelah pelatihan:")
print(lapisan2)
