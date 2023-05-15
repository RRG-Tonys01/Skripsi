import pandas as pd
import numpy as np


class CSVReader:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def change_data_type(self, date_col='Tanggal', val_col='Terakhir'):
        self.df[date_col] = pd.to_datetime(
            self.df[date_col], format="%d/%m/%Y")
        # self.df[val_col] = self.df[val_col].str.replace(
        #     '.', '').str.replace(',', '.').astype(float)
        if val_col in self.df.columns and self.df[val_col].dtype == 'object':
            self.df[val_col] = self.df[val_col].str.replace(
                '.', '').str.replace(',', '.').astype(float)

        self.df = self.df[[date_col, val_col]]

    def get_data(self):
        return self.df


class MergeData:
    def __init__(self, emas_data, ihsg_data, minyak_mentah_data, kurs_data):
        self.emas_data = emas_data
        self.ihsg_data = ihsg_data
        self.minyak_mentah_data = minyak_mentah_data
        self.kurs_data = kurs_data

    def merge_data(self):
        mergedData = pd.merge(self.emas_data, self.ihsg_data, on='Tanggal')
        mergedData = pd.merge(
            mergedData, self.minyak_mentah_data, on='Tanggal')
        mergedData = mergedData.rename(columns={'Terakhir_x': 'Emas',
                                                'Terakhir_y': 'IHSG',
                                                'Terakhir': 'Minyak Mentah'})
        mergedData = pd.merge(mergedData, self.kurs_data, on='Tanggal')
        mergedData = mergedData.rename(columns={'Terakhir': 'Kurs USD/IDR'})

        mergedData['Minyak Mentah'] = mergedData['Minyak Mentah'].fillna(
            0) * mergedData['Kurs USD/IDR'].fillna(0)
        mergedData['Minyak Mentah'] = mergedData['Minyak Mentah'].replace(
            0.000, np.NaN)

        # Return nilai dari Merged Data Tanpa proses Cleaing Data
        # return mergedData

        # Menghapus data yang bernilai NaN
        merged_data_clean = mergedData.dropna()
        return merged_data_clean
