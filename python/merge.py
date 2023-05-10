# Menggabungkan emasData, ihsgData, minyakMentahData, dan kursData
mergedData = pd.merge(emasData, ihsgData, on='Tanggal')
mergedData = pd.merge(mergedData, minyakMentahData, on='Tanggal')
mergedData = pd.merge(mergedData, kursData, on='Tanggal')

mergedData = mergedData.rename(columns={'Terakhir_x': 'Emas',
                                        'Terakhir_y': 'IHSG',
                                        'Terakhir_x': 'Minyak Mentah',
                                        'Terakhir_y': 'Kurs USD/IDR'})

# mengubah kolom 'Tanggal' menjadi tipe data datetime
mergedData['Tanggal'] = pd.to_datetime(
    mergedData['Tanggal'], format='%d/%m/%Y')
mergedData = mergedData.sort_values(by=['Tanggal'])

# mengubah format tanggal menjadi "tanggal/bulan/tahun"
mergedData['Tanggal'] = mergedData['Tanggal'].dt.strftime('%d/%m/%Y')


# Menampilkan hasil gabungan data
# print(mergedData)
mergedData['Minyak Mentah'].isna()
