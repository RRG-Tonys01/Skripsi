import pandas as pd
from date_range import DateRange
from csvreader import CSVReader


# Date Range
dr = DateRange("01/01/2018", "31/03/2023")
date_array = dr.get_date_array()

# membuat objek CSVReader dan membaca file CSV
emasDataset = CSVReader(
    "../dataset/Data Historis Emas Berjangka.csv").read_csv()
ihsgDataset = CSVReader(
    "../dataset/Data Historis Emas Berjangka.csv").read_csv()
minyakMentahDataset = CSVReader(
    "../dataset/Data Historis Emas Berjangka.csv").read_csv()
kursDataset = CSVReader(
    "../dataset/Data Historis Emas Berjangka.csv").read_csv()

# print(emasDataset.read_csv())
emasData = emasDataset[['Tanggal', 'Terakhir']]
# ihsgData = ihsgDataset[['Tanggal', 'Terakhir']]
# minyakMentahData = minyakMentahDataset[['Tanggal', 'Terakhir']]
# kursData = kursDataset[['Tanggal', 'Terakhir']]

# print("======== Data Historis Emas ========")
# print(emasData)
# print("======== Data Historis IHSG ========")
# print(ihsgData)
# print("======== Data Historis Minyak Mentah ========")
# print(minyakMentahData)
# print("======== Data Historis Kurs Rupiah ========")
# print(kursData)
