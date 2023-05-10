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

# Membersihkan data
emas_dataset.change_data_type()
ihsg_dataset.change_data_type()
minyak_mentah_dataset.change_data_type()
kurs_dataset.change_data_type()

# Mendapatkan data
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

# Describe nilai Statistika, seperti  : count, mean, std, min, 25%, 50%, 75%, dan max
# print(clean_data.describe())

# Memberikan informasi terhadap setiap kolom pada data
# print(clean_data.info())

# Membuat Korelasi attribute terhadap keterkaitan antar variable dengan kenaikan harga emas
# print(clean_data.corr())
# sns.heatmap(clean_data.corr(), annot=True, cmap='coolwarm')
# plt.show()

# print("======== Data Historis Emas ========")
# print(emas_data)
# print("======== Data Historis IHSG ========")
# print(ihsg_data)
# print("======== Data Historis Minyak Mentah ========")
# print(minyak_mentah_data)
# print("======== Data Historis Kurs Rupiah ========")
# print(kurs_data)


####################################################################
###                                                              ###
###         PEMODELAN ALGORITMA LONG-SHORT TERM MEMORY           ###
###                         (LSTM)                               ###
###                                                              ###
####################################################################

scaler = MinMaxScaler(feature_range=(0, 1))
data = clean_data['Emas'].values.reshape(-1, 1)
scaled_data = scaler.fit_transform(data)
# print(scaled_data)
# print(len(scaled_data))

# Membagi data menjadi data latih dan data uji dengan perbandingan 80:20.
training_size = int(len(scaled_data) * 0.8)
testing_size = len(scaled_data) - training_size
training_data = scaled_data[0:training_size, :]
testing_data = scaled_data[training_size:len(scaled_data), :]
# print("Jumlah Data Latih : " + len(training_data))
# print("Jumlah Data Uji : " + len(testing_data))
# print("Jumlah Data Menyeluruh : " + len(scaled_data))


def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)


# Pengjuian nilai time_step adalah 50,100, 150, dan 200
time_step = 100
X_train, Y_train = create_dataset(training_data, time_step)
X_test, Y_test = create_dataset(testing_data, time_step)
print(len(X_train), len(Y_train))
print(len(X_test), len(Y_test))

# Membuat model LSTM dengan menggunakan library tensorflow.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(
    50, return_sequences=True, input_shape=(time_step, 1)))
model.add(tf.keras.layers.LSTM(50, return_sequences=True))
model.add(tf.keras.layers.LSTM(50))
model.add(tf.keras.layers.Dense(1))

# Evaluasi Menggunakan MSE dan RMSE
model.compile(loss='mean_squared_error', optimizer='adam')

# Evaluasi Menggunakan MAE
# model.compile(loss='mean_absolute_error', optimizer='adam')


# Melakukan pelatihan (training) pada model dengan data latih yang telah dibuat.
model.fit(X_train, Y_train, validation_data=(
    X_test, Y_test), epochs=100, batch_size=64, verbose=1)

# Melakukan prediksi pada data uji dengan menggunakan model yang telah dilatih.
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# Melakukan invers normalisasi pada hasil prediksi dan data sebenarnya.
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])


# Menghitung nilai rata-rata error pada data latih dan data uji.
rmse_train = np.sqrt(mean_squared_error(Y_train[0], train_predict[:, 0]))
rmse_test = np.sqrt(mean_squared_error(Y_test[0], test_predict[:, 0]))
print("RMSE train: ", rmse_train)
print("RMSE test: ", rmse_test)
