import csv
import glob
import os


def get_path():

    list_of_files = glob.glob('./data/*') # * means all if need specific format then *.csv
    print(list_of_files)
    latest_file = max(list_of_files, key=os.path.getctime)

    path = []

    with open(latest_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = -1
        for row in csv_reader:
            if line_count == -1:
                line_count += 1
            elif line_count == 0:
                line_count += 1
            else:
                path.append([float(row[2]),float(row[3])])
                line_count += 1
    return path