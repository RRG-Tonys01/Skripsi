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
data = clean_data[['Emas', 'IHSG', 'Minyak Mentah', 'Kurs USD/IDR']]
scaled_data = scaler.fit_transform(data.values)


# 4. Membuat dataset dengan time step
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step):
        a = data[i:(i+time_step), :]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)


time_step = 50
X, Y = create_dataset(scaled_data, time_step)

# 5. Membagi dataset menjadi training dan testing set
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

# 6. Membuat model LSTM
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(
    50, return_sequences=True, input_shape=(time_step, 4)))
model.add(tf.keras.layers.LSTM(50, return_sequences=True))
model.add(tf.keras.layers.LSTM(50))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 7. Melatih model
model.fit(X_train, Y_train, validation_data=(
    X_test, Y_test), epochs=50, batch_size=64, verbose=1)

# 8. Melakukan prediksi
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 9. Mengembalikan data ke skala semula
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform(Y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))


# 10. Menghitung nilai error
rmse_train = np.sqrt(mean_squared_error(Y_train[0], train_predict[:, 0]))
rmse_test = np.sqrt(mean_squared_error(Y_test[0], test_predict[:, 0]))
print("MSE train : ", mean_squared_error(Y_train[0], train_predict[:, 0]))
print("MSE test : ", mean_squared_error(Y_test[0], test_predict[:, 0]))
print("RMSE train: ", rmse_train)
print("RMSE test: ", rmse_test)

# 11. Plot hasil prediksi
plt.plot(Y_test[0], label='Actual')
plt.plot(test_predict, label='Predicted')
plt.legend()
plt.show()
