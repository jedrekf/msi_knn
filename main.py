import configparser
import csv
import math
import operator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import visualize as V

def load_data(path, test_ratio):
    with open(path, 'r') as csvfile:
        data_reader = csv.reader(csvfile)
        data = []
        test_data = []
        skip = int(1 / test_ratio)
        print(skip)
        i = 0
        for row in data_reader:
            i = i + 1
            if i == 1:
                continue

            if i % skip == 0:
                test_data.append((float(row[0]), float(row[1]), int(row[2])))
            else:
                data.append((float(row[0]), float(row[1]), int(row[2])))
        return data, test_data


def calc_dist(instance1, instance2):
    distance = 0
    distance += pow((instance1[0] - instance2[0]), 2)
    distance += pow((instance1[1] - instance2[1]), 2)
    return math.sqrt(distance)


def knn(train_data, instance, k):
    distances = []
    for train_row in train_data:
        dist = calc_dist(train_row, instance)
        distances.append((train_row[2], dist))

    distances.sort(key=lambda x: x[1])
    neighbours = distances[0:k]

    classes = {}
    for neighbour in neighbours:
        if neighbour[0] in classes:
            classes[neighbour[0]] += 1
        else:
            classes[neighbour[0]] = 1

    return  max(classes.items(), key=operator.itemgetter(1))[0]


def test(train_data, test_data, k):
    correct_answers = 0
    wrong_answers = 0
    for test_instance in test_data:
        answer_class = test_instance[2]
        ret_class = knn(train_data, (test_instance[0], test_instance[1]), k)
        if answer_class == ret_class:
            correct_answers += 1
        else:
            wrong_answers += 1
    
    print("Correct: {}, Wrong: {}".format(correct_answers, wrong_answers))


def main():
    config = configparser.ConfigParser()

    if len(config.read("config")) == 0:
        print("Configuration file was not found!\n")
        return

    params = config["PARAMS"]
    train_file = str(params["train_file"])
    test_ratio = float(params["test_ratio"])
    k = int(params["k"])
    mesh_step = float(params["mesh_step"])
    do_validation = bool(params["do_validation"])
    do_visulization = bool(params["do_visulization"])
    
    print("{},\n{},\n{}".format(train_file, test_ratio, k))

    train_data, test_data = load_data(train_file, test_ratio)

    print(len(test_data), len(train_data))

    if do_validation:
        test(train_data, test_data, k)

    if do_visulization:
        #this will draw classes and plot testing points
        V.visualize(train_data, test_data, k, mesh_step);


if __name__ == '__main__':
    main()
