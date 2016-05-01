import csv, sys, os
import operator
from emails import get_emails

def sort_csv(csv_filename, output):
    """sort (and rewrite) a csv file.
    types:  data types (conversion functions) for each column in the file
    sort_key_columns: column numbers of columns to sort by"""
    data = []
    with open(csv_filename, 'r') as f:
        for row in csv.reader(f):
            data.append(row)
    data.sort(key=operator.itemgetter(0), reverse=True)
    with open(output, 'w') as f:
        csv.writer(f).writerows(data)

if __name__ == "__main__":
    subjects = get_emails("subjects.txt")
    for subject in subjects:
        in_file = os.path.join("benchmark_set", subject + ".txt")
        out_file = os.path.join("benchmark_set", subject + ".txt")
        sort_csv(in_file, out_file)
    # sort_csv(in_file, out_file)
