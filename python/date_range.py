import datetime


class DateRange:
    def __init__(self, start_date_str, end_date_str):
        self.start_date = datetime.datetime.strptime(
            start_date_str, "%d/%m/%Y").date()
        self.end_date = datetime.datetime.strptime(
            end_date_str, "%d/%m/%Y").date()

    def get_date_array(self):
        delta = datetime.timedelta(days=1)
        date_array = []
        while self.start_date <= self.end_date:
            date_array.append(self.start_date.strftime("%d/%m/%Y"))
            self.start_date += delta
        return date_array
