class Euclidean:

    @staticmethod
    def calc_dist(instance1, instance2):
        distance = 0
        distance += pow((instance1[0] - instance2[0]), 2)
        distance += pow((instance1[1] - instance2[1]), 2)
        return distance
