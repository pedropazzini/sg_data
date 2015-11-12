from scipy.spatial import distance
import numpy as np

# Code from https://www.snip2code.com/Snippet/76076/Fr-chet-Distance-in-Python
class Frechet_distance:

    def __init__(self, P, Q):

        self.P = P
        self.Q = Q


    def euc_dist(self,pt1,pt2):
        return distance.euclidean(pt1,pt2)

    def _c(self,ca,i,j,P,Q):
        if ca[i,j] > -1:
            return ca[i,j]
        elif i == 0 and j == 0:
            ca[i,j] = self.euc_dist(P[0],Q[0])
        elif i > 0 and j == 0:
            ca[i,j] = max(self._c(ca,i-1,0,P,Q),self.euc_dist(P[i],Q[0]))
        elif i == 0 and j > 0:
            ca[i,j] = max(self._c(ca,0,j-1,P,Q),self.euc_dist(P[0],Q[j]))
        elif i > 0 and j > 0:
            ca[i,j] = max(min(self._c(ca,i-1,j,P,Q),self._c(ca,i-1,j-1,P,Q),self._c(ca,i,j-1,P,Q)),self.euc_dist(P[i],Q[j]))
        else:
            ca[i,j] = float("inf")
        return ca[i,j]

    """ Computes the discrete frechet distance between two polygonal lines
    Algorithm: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
    P and Q are arrays of 2-element arrays (points)
    """
    def frechetDist(self):
        P = self.P
        Q = self.Q
        ca = np.ones((len(P),len(Q)))
        ca = np.multiply(ca,-1)
        return self._c(ca,len(P)-1,len(Q)-1,P,Q)
