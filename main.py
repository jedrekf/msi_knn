import configparser
import csv
import operator
import time

import knn as knn
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


def test(train_data, test_data, k, do_manhattan):
    all_ret_classes = []
    correct_answers = 0
    wrong_answers = 0

    knn_instance = knn.KNN(train_data, k, do_manhattan)
    for test_instance in test_data:
        answer_class = test_instance[2]
        ret_class = knn_instance.compute_class((test_instance[0], test_instance[1]))
        all_ret_classes.append(ret_class)
        if answer_class == ret_class:
            correct_answers += 1
        else:
            wrong_answers += 1

    print("Correct: {}, Wrong: {}".format(correct_answers, wrong_answers))

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
    do_validation = config.getboolean(section_name, "do_validation")
    do_visulization = config.getboolean(section_name, "do_visulization")

    print("train file {},\ntest ratio {},\nk parameter {}".format(train_file, test_ratio, k))

    train_data, test_data = load_data(train_file, test_ratio)

    num_of_classes = max(test_data, key=operator.itemgetter(-1))[-1]
    start = time.time()
    print("Length of test data: {}, train data: {}".format(len(test_data), len(train_data)))
    different_results = []

    result = test(train_data, test_data, k, do_manhattan)
    guessed_classes = result[0]
    accuracy = result[1]
    # different_results.append((guessed_classes, accuracy, k))

    best_result = result  # max(different_results, key=operator.itemgetter(1))

    if do_validation:
        print(best_result[1])

    end = time.time() - start
    print('end time' + str(end))

    if do_visulization:
        # this will draw classes and plot testing points
        V.visualize(best_result[0], train_data, test_data, k, num_of_classes, mesh_step, do_manhattan)


if __name__ == '__main__':
    main()
