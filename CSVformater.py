import csv

filename = 'winemag-data-130k-v2.csv'
with open(filename, 'r', encoding="utf8") as csvfile:
    w = csv.writer(open("catXYdata.csv", "w", encoding="utf8"))
    datareader = csv.reader(csvfile)

    for row in datareader:
        if row[4] == "points":  #För att inte formatera första raden i CSV filen
            rate = "points"
        else:
            rate = int(row[4])
            if 80 <= rate < 85:
                rate = str("OK")
            elif 85 <= rate < 90:
                rate = str("Good")
            elif 90 <= rate < 95:
                rate = str("Tasty")
            else:
                rate = str("Perfect")

        w.writerow([row[2], rate])







