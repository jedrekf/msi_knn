import configparser
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from main import load_data, knn


def visualize(all_guessed_classes, train_data, test_data, num_of_classes, mesh_step):
    # colors for visualization - should be generated based on number of classes

    cmap_bold, cmap_light = generate_colors_per_class(num_of_classes)

    all_x = [row[0] for row in train_data]
    all_y = [row[1] for row in train_data]
    # get surface boudaries for plotting
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    # create mesh points from boundary values
    xx = np.arange(x_min, x_max, mesh_step)
    yy = np.arange(y_min, y_max, mesh_step)
    np.append(xx, x_max)
    np.append(yy, y_max)
    # create mesh for x's and y's
    xx, yy = np.meshgrid(xx, yy)

    point_classes = []
    for x, y in zip(xx.ravel(), yy.ravel()):
        mesh_point_class = knn(train_data, (x, y), 3)
        point_classes.append(mesh_point_class)

    point_classes = np.reshape(point_classes, xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, point_classes, cmap=cmap_light)

    # add drawing test points
    plt.scatter([xVal[0] for xVal in test_data], [yVal[1] for yVal in test_data],
                c=all_guessed_classes, cmap=cmap_bold, edgecolor='k', s=20)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Visualize it.")
    plt.show()


def generate_colors_per_class(num_of_classes):
    cmap_light = []
    cmap_bold = []
    for i in range(num_of_classes):
        r = lambda: random.randint(0, 255)
        color = ('#%02X%02X%02X' % (r(), r(), r()))
        cmap_light.append(color)
    for i in range(len(cmap_light)):
        cmap_bold.append(colorscale(cmap_light[i], 1.5))
    cmap_light = ListedColormap(cmap_light)
    cmap_bold = ListedColormap(cmap_bold)
    return cmap_bold, cmap_light


def clamp(val, minimum=0, maximum=255):
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return val


def colorscale(hexstr, scalefactor):
    hexstr = hexstr.strip('#')

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    r = int(clamp(r * scalefactor))
    g = int(clamp(g * scalefactor))
    b = int(clamp(b * scalefactor))

    return "#%02x%02x%02x" % (r, g, b)


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

    print("train file {},\ntest ratio {},\nk parameter {}".format(train_file, test_ratio, k))

    train_data, test_data = load_data(train_file, test_ratio)

    print(len(test_data), len(train_data))

    visualize(train_data, test_data, k, mesh_step)


if __name__ == '__main__':
    main()
