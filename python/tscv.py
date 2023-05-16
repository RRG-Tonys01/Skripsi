import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from objectClass import *
from function import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# Mengabaikan semua Warning pada Output
warnings.filterwarnings('ignore')


# mengatur opsi pandas untuk menampilkan seluruh kolom dan baris
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# Buat rentang tanggal yang akan diperiksa
start_date = '2018-01-01'
end_date = '2023-03-31'

# Membuat objek CSVReader dan membaca file CSV
emas_dataset = CSVReader("../dataset/Data Historis Emas Berjangka.csv")
ihsg_dataset = CSVReader(
    "../dataset/Data Historis Jakarta Stock Exchange Composite.csv")
minyak_mentah_dataset = CSVReader(
    "../dataset/Data Historis Minyak Mentah WTI Berjangka.csv")
kurs_dataset = CSVReader("../dataset/Data Historis USD_IDR.csv")

# # Mengubah tipe data
emas_dataset.change_data_type()
ihsg_dataset.change_data_type()
minyak_mentah_dataset.change_data_type()
kurs_dataset.change_data_type()

# # Mendapatkan data
emas_data = emas_dataset.get_data()
ihsg_data = ihsg_dataset.get_data()
minyak_mentah_data = minyak_mentah_dataset.get_data()
kurs_data = kurs_dataset.get_data()

# Menambahkan tanggal yang tidak termasuk ke dalam data
emas_data, ihsg_data, minyak_mentah_data, kurs_data = fill_missing_dates(
    start_date, end_date, emas_data, ihsg_data, minyak_mentah_data, kurs_data)

# urutkan DataFrame berdasarkan tanggal
emas_data = emas_data.sort_values(by='Tanggal')
ihsg_data = ihsg_data.sort_values(by='Tanggal')
minyak_mentah_data = minyak_mentah_data.sort_values(by='Tanggal')
kurs_data = kurs_data.sort_values(by='Tanggal')

# Penggabungan seluruh dataset
mergedData = MergeData(emas_data, ihsg_data, minyak_mentah_data, kurs_data)
clean_data = mergedData.merge_data()

# print(clean_data)
# clean_data.to_csv('../dataset/Prediksi Harga Emas.csv', index=False)

# Describe nilai Statistika, seperti  : count, mean, std, min, 25%, 50%, 75%, dan max
# print(clean_data.describe())

# Memberikan informasi terhadap setiap kolom pada data
# print(clean_data.info())

# Membuat Korelasi attribute terhadap keterkaitan antar variable dengan kenaikan harga emas
# print(clean_data.corr())
# sns.heatmap(clean_data.corr(), annot=True, cmap='coolwarm')
# plt.show()


####################################################################
###                                                              ###
###         PEMODELAN ALGORITMA LONG-SHORT TERM MEMORY           ###
###                         (LSTM)                               ###
###                                                              ###
####################################################################

# Normalisasi data menggunakan MinMaxScaler
scaler = MinMaxScaler()
data = clean_data[['Emas', 'IHSG', 'Minyak Mentah', 'Kurs USD/IDR']]
scaled_data = scaler.fit_transform(data)

# Buat fungsi untuk membuat dataset dengan time steps


def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data)-time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)


# Buat dataset dengan time steps sebesar 3
time_steps = 50
X, y = create_dataset(scaled_data, time_steps)

# Split dataset menggunakan TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)


mse_scores = []
rmse_scores = []
mae_scores = []
prediction_scores = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Buat model LSTM
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(64, activation='relu', input_shape=(
        time_steps, X_train.shape[2])))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(X_train.shape[2]))
    model.compile(optimizer='adam', loss='mse')
    # model.compile(optimizer='adam', loss='mae')

    # Training model
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)

    # Evaluasi model
    mse = model.evaluate(X_test, y_test)
    rmse = np.sqrt(mse)
    # mae = model.evaluate(X_test, y_test)

    mse_scores.append(mse)
    rmse_scores.append(rmse)
    # mae_scores.append(mae)

    # Prediksi harga emas untuk waktu ke-1 hingga waktu ke-3 di masa depan
    last_X = scaled_data[-time_steps:]
    last_X = last_X.reshape((1, time_steps, X_train.shape[2]))
    prediction = model.predict(last_X)

    # Invers normalisasi data
    prediction = scaler.inverse_transform(prediction)
    y_test = scaler.inverse_transform(y_test)

    prediction_scores.append(prediction[0][0])

# Print hasil evaluasi dari k-Fold Cross Validation
print('MSE scores:', mse_scores)
print('RMSE scores:', rmse_scores)
# print('MAE scores:', mae_scores)
print('Rata-rata MSE:', np.mean(mse_scores))
print('Rata-rata RMSE:', np.mean(rmse_scores))
# print('Rata-rata MAE:', np.mean(mae_scores))

# Print prediksi harga emas
print('Prediksi harga emas untuk waktu ke-1 hingga waktu ke-3 di masa depan:')
print(prediction_scores)
