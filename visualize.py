import configparser
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from main import load_data, knn

def visualize(train_data, test_data, k, mesh_step):
    #colors for visualization - should be generated based on number of classes
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    all_x = [row[0] for row in train_data]
    all_y = [row[1] for row in train_data]
    #get surface boudaries for plotting
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    # create mesh points from boundary values
    xx = np.arange(x_min, x_max, mesh_step);
    yy = np.arange(y_min, y_max, mesh_step);
    np.append(xx, x_max);
    np.append(yy, y_max);
    #create mesh for x's and y's
    xx, yy = np.meshgrid(xx, yy)

    point_classes = []
    for x, y in zip(xx.ravel(), yy.ravel()):
        mesh_point_class = knn(train_data, (x, y), k)
        point_classes.append(mesh_point_class)

    test_point_classes = []
    for x, y in zip([row[0] for row in test_data], [row[1] for row in test_data]):
        mesh_point_class = knn(train_data, (x, y), k)
        test_point_classes.append(mesh_point_class)

    point_classes = np.reshape(point_classes, xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, point_classes, cmap=cmap_light)

    # add drawing test points
    plt.scatter([row[0] for row in test_data], [row[1] for row in test_data],
        c=test_point_classes, cmap=cmap_bold, edgecolor='k', s=20)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Visualize it.")
    plt.show()

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

    print("{},\n{},\n{}".format(train_file, test_ratio, k))

    train_data, test_data = load_data(train_file, test_ratio)

    print(len(test_data), len(train_data))

    visualize(train_data, test_data, k, mesh_step)

if __name__ == '__main__':
    main()
