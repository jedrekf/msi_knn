import configparser
import csv
import operator
import time

import czebyszew as czebyszew
import euclidean as euclidean
import knn as knn
import manhattan as manhattan
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


def test(train_data, test_data, k, metric):
    all_ret_classes = []
    correct_answers = 0
    wrong_answers = 0

    knn_instance = knn.KNN(train_data, k, metric)
    for test_instance in test_data:
        answer_class = test_instance[2]
        ret_class = knn_instance.compute_class((test_instance[0], test_instance[1]))
        all_ret_classes.append(ret_class)
        if answer_class == ret_class:
            correct_answers += 1
        else:
            wrong_answers += 1

    print("{};{}".format(k, correct_answers / (correct_answers + wrong_answers)))

    return all_ret_classes, correct_answers / (correct_answers + wrong_answers)


def main():
    config = configparser.ConfigParser()

    if len(config.read("config")) == 0:
        print("Configuration file was not found!\n")
        return

    section_name = "PARAMS"

    params = config[section_name]
    train_file = str(params["train_file"])
    test_ratio = float(params["test_ratio"])
    k = int(params["k"])
    mesh_step = float(params["mesh_step"])

    do_manhattan = config.getboolean(section_name, "use_manhattan")
    do_czebyszew = config.getboolean(section_name, "use_Czebyszew")
    do_validation = config.getboolean(section_name, "do_validation")
    do_visulization = config.getboolean(section_name, "do_visulization")

    print("train file {},\ntest ratio {},\nk parameter {}".format(train_file, test_ratio, k))

    train_data, test_data = load_data(train_file, test_ratio)

    num_of_classes = max(test_data, key=operator.itemgetter(-1))[-1]
    start = time.time()
    print("Length of test data: {}, train data: {}".format(len(test_data), len(train_data)))

    k_vals = [1, 5, 10, 20, 25, 50, 100]
    if do_czebyszew:
        final_metric = czebyszew.Czebyszew.calc_dist
    elif do_manhattan:
        final_metric = manhattan.Manhattan.calc_dist
    else:
        final_metric = euclidean.Euclidean.calc_dist

    result = None
    for i in k_vals:
        start = time.time()
        result = test(train_data, test_data, i, final_metric)
        end = time.time() - start
        print(';' + str(end))

    best_result = result

    end = time.time() - start
    print('end time' + str(end))

    if do_visulization:
        # this will draw classes and plot testing points
        V.visualize(best_result[0], train_data, test_data, k, num_of_classes, mesh_step, do_manhattan)


if __name__ == '__main__':
    main()
