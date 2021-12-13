import csv
import json

filename = 'winemag-data-130k-v2.csv'

with open(filename, 'r', encoding="utf8") as csvfile:
    w = csv.writer(open("XYdata.csv", "w", encoding="utf8"))
    count=0
    datareader = csv.reader(csvfile)
    for row in datareader:
        #print(row)
        #print(row[2])
        if count == 0:
            pass
        else:
            w.writerow([row[2], row[4]])
        count += 1





