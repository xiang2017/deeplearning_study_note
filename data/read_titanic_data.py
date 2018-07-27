import csv
import numpy as np


f = __file__
file_name = f[:f.rfind('/')] + '/titanic/train.csv'

carbin_map = {
    "": 0,
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "T": 8,
}

embarked_map = {
    "": 0,
    "S": 1,
    "C": 2,
    "Q": 3
}


def load_dataset():
    csv_reader = csv.reader(open(file_name, encoding='utf-8'))
    x = []
    y = []
    first = True
    for row in csv_reader:
        if first:
            first = False
            continue

        pclass = int(row[2])
        sex = 1 if row[4] == 'male' else 2
        age = float(row[5] if row[5] != '' else 35)
        sibsp = int(row[6])
        parch = int(row[7])
        fare = float(row[9])
        carbin = carbin_map[row[10][0]] if row[10] != '' else 0
        embarked = embarked_map[row[11]]
        x.append([pclass, sex, age, sibsp, parch, fare, carbin, embarked])
        y.append(row[1])

    x = np.array(x, dtype=float).T
    y = np.array(y, dtype=float).reshape((1, len(y)))
    return x, y




if __name__ == '__main__':
    file_name = 'titanic/train.csv'
    load_dataset()