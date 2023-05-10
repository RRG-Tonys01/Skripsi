import pandas as pd
import numpy as np


def fill_missing_dates(start_date, end_date, emas_data, ihsg_data, minyak_mentah_data, kurs_data):
    all_dates = pd.DataFrame(columns=emas_data.columns)
    missing_dates = pd.date_range(
        start=start_date, end=end_date).difference(emas_data['Tanggal'])
    all_dates = all_dates.append(pd.DataFrame({'Tanggal': missing_dates}))
    all_dates = all_dates.fillna(value=pd.np.nan)
    emas_data = pd.concat([emas_data, all_dates], ignore_index=True)

    all_dates = pd.DataFrame(columns=ihsg_data.columns)
    missing_dates = pd.date_range(
        start=start_date, end=end_date).difference(ihsg_data['Tanggal'])
    all_dates = all_dates.append(pd.DataFrame({'Tanggal': missing_dates}))
    all_dates = all_dates.fillna(value=pd.np.nan)
    ihsg_data = pd.concat([ihsg_data, all_dates], ignore_index=True)

    all_dates = pd.DataFrame(columns=minyak_mentah_data.columns)
    missing_dates = pd.date_range(start=start_date, end=end_date).difference(
        minyak_mentah_data['Tanggal'])
    all_dates = all_dates.append(pd.DataFrame({'Tanggal': missing_dates}))
    all_dates = all_dates.fillna(value=pd.np.nan)
    minyak_mentah_data = pd.concat(
        [minyak_mentah_data, all_dates], ignore_index=True)

    all_dates = pd.DataFrame(columns=kurs_data.columns)
    missing_dates = pd.date_range(
        start=start_date, end=end_date).difference(kurs_data['Tanggal'])
    all_dates = all_dates.append(pd.DataFrame({'Tanggal': missing_dates}))
    all_dates = all_dates.fillna(value=pd.np.nan)
    kurs_data = pd.concat([kurs_data, all_dates], ignore_index=True)

    return emas_data, ihsg_data, minyak_mentah_data, kurs_data
