import pandas as pd


class CSVReader:
    def __init__(self, filename):
        self.filename = filename

    def read_csv(self):
        df = pd.read_csv(self.filename)
        return df
