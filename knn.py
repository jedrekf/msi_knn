import operator


class KNN:
    def __init__(self, train_data, k, use_manhattan):

        self.train_data = train_data
        self.k = k
        self.use_manhattan = use_manhattan

    def compute_class(self, instance):
        distances = self.measure_distances(instance)
        neighbours = distances[0:self.k]

        classes = {}
        for neighbour in neighbours:
            if neighbour[0] in classes:
                classes[neighbour[0]] += 1
            else:
                classes[neighbour[0]] = 1

        return max(classes.items(), key=operator.itemgetter(1))[0]

    def measure_distances(self, instance):
        distances = []
        for train_row in self.train_data:
            dist = self.calc_dist_manhattan(train_row, instance) if self.use_manhattan else self.calc_dist_euclidean(
                train_row, instance)
            distances.append((train_row[2], dist))
        distances.sort(key=lambda x: x[1])
        return distances

    def calc_dist_euclidean(self, instance1, instance2):
        distance = 0
        distance += pow((instance1[0] - instance2[0]), 2)
        distance += pow((instance1[1] - instance2[1]), 2)
        return distance

    def calc_dist_manhattan(self, instance1, instance2):
        distance = 0
        distance += abs(instance1[0] - instance2[0])
        distance += abs(instance1[1] - instance2[1])
        return distance
