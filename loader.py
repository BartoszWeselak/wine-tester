import csv
from tkinter import filedialog
def read_csv_file():
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    data = []
    if filename:
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                data.append(row)
        if data:
            print("file loaded succesfuly.")
        else:
            print("file is empty.")

    else:
        print("file not selected.")
    return data