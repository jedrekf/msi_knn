class Czebyszew:
    @staticmethod
    def calc_dist(instance1, instance2):
        distance1 = abs(instance1[0] - instance2[0])
        distance2 = abs(instance1[1] - instance2[1])
        return max(distance1, distance2)
