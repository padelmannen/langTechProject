import csv

filename = 'winemag-data-130k-v2.csv'

with open(filename, 'r', encoding="utf8") as csvfile:
    w = csv.writer(open("XYdata.csv", "w", encoding="utf8"))
    datareader = csv.reader(csvfile)
    for row in datareader:
        w.writerow([row[2], row[4]])






