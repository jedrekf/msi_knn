class Manhattan:
    @staticmethod
    def calc_dist(instance1, instance2):
        distance = 0
        distance += abs(instance1[0] - instance2[0])
        distance += abs(instance1[1] - instance2[1])
        return distance
