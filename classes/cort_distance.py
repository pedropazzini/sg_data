import numpy as np

# Based on the paper http://eprints.uni-mysore.ac.in/3921/1/10.1007_s11634-006-0004-6

class Cort_distance:

    def __init__(self, s1, s2, k = 1.5, delta_conv = None):

        self.S1 = s1
        self.S2 = s2

        self.k = k

        self.delta_conv = delta_conv

    def cort(self):

        size = len(self.S1)

        if (size != len(self.S2)):
            raise ValueError("Time series with different lengths,,,")

        n = 0
        d1 = 0
        d2 = 0
        for i in range(1,size):
            s1 = self.S1[i] - self.S1[i-1]
            s2 = self.S2[i] - self.S2[i-1]
            n += s1*s2

            d1 += s1**2
            d2 += s2**2

        return n/(np.sqrt(d1)*np.sqrt(d2))

    def tunning_function(self):
        return 2/(1 + np.exp(self.k*self.cort()))

    def get_dissimilarity_index(self):
        return self.tunning_function() * (self.delta_conv if self.delta_conv is not None else 1)




