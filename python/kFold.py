import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import KFold
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
data = clean_data[['Emas', 'IHSG', 'Minyak Mentah', 'Kurs USD/IDR']].values
scaled_data = scaler.fit_transform(data)
# print(scaled_data)

# Membagi data menjadi data latih dan data uji dengan perbandingan 80:20.
training_size = int(len(scaled_data) * 0.8)
testing_size = len(scaled_data) - training_size
training_data = scaled_data[0:training_size, :]
testing_data = scaled_data[training_size:len(scaled_data), :]


# Membuat dataset terbaru
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)


# Menambahkan data IHSG, Minyak Mentah, dan Kurs USD/IDR ke dalam dataset
time_step = 50
X, Y = create_dataset(scaled_data, time_step)
data_X = np.zeros((X.shape[0], X.shape[1], 4))
data_X[:, :, 0] = X[:, 0][:, np.newaxis]
data_X[:, :, 1] = scaled_data[50:1280, 1][:, np.newaxis]
data_X[:, :, 2] = scaled_data[50:1280, 2][:, np.newaxis]
data_X[:, :, 3] = scaled_data[50:1280, 3][:, np.newaxis]

# print("=============== Transformasi Data ===============")
# print(scaled_data)
# print("================== ----------- ==================")
# print("================== New Dataset ==================")
# print(X)
# print("================== ----------- ==================")
# print("=========  Penambahan Variable ke Dalam =========")
# print("================ Dataset Terbaru ================")
# print(data_X)
# print("================== ----------- ==================")

# Membuat model LSTM dengan menggunakan library tensorflow.
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 4)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Melakukan cross-validation
kf = KFold(n_splits=5)
mse_scores = []
rmse_scores = []
for train_index, test_index in kf.split(data_X):
    X_train, X_test = data_X[train_index], data_X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    model.fit(X_train, Y_train, epochs=50, batch_size=64, verbose=1)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    Y_train = scaler.inverse_transform([Y_train])
    test_predict = scaler.inverse_transform(test_predict)
    Y_test = scaler.inverse_transform([Y_test])
    mse_train = mean_squared_error(Y_train[0], train_predict[:, 0])
    mse_test = mean_squared_error(Y_test[0], test_predict[:, 0])
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    mse_scores.append(mse_test)
    rmse_scores.append(rmse_test)

# # Menampilkan hasil evaluasi
# print("MSE scores:", mse_scores)
# print("RMSE scores:", rmse_scores)
# print("Average RMSE:", np.mean(rmse_scores))
