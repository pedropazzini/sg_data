import numpy as np


class DTW:

    def __init__(self, s1, s2):

        self.s1 = s1
        self.s2 = s2

    def get(self,is_LB_keogh=False,w=None,r=5):

        if (is_LB_keogh):
            return self.LB_Keogh(self.s1,self.s2,r)
        else:
            return self.DTWDistance(self.s1,self.s2,w)

    def DTWDistance(self,s1,s2,w=None):
        '''
        Calculates dynamic time warping Euclidean distance between two
        sequences. Option to enforce locality constraint for window w.
        '''
        DTW={}

        if w:
            w = max(w, abs(len(s1)-len(s2)))
            for i in range(-1,len(s1)):
                for j in range(-1,len(s2)):
                    DTW[(i, j)] = float('inf')
        else:
            for i in range(len(s1)):
                DTW[(i, -1)] = float('inf')
            for i in range(len(s2)):
                DTW[(-1, i)] = float('inf')

            DTW[(-1, -1)] = 0

        for i in range(len(s1)):
            if w:
                for j in range(max(0, i-w), min(len(s2), i+w)):
                    dist= (s1[i]-s2[j])**2
                    DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            else:
                for j in range(len(s2)):
                    dist= (s1[i]-s2[j])**2
                    DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

        return np.sqrt(DTW[len(s1)-1, len(s2)-1])

    def LB_Keogh(self,s1,s2,r):
        '''
        Calculates LB_Keough lower bound to dynamic time warping. Linear
        complexity compared to quadratic complexity of dtw.
        '''
        LB_sum=0
        for ind,i in enumerate(s1):
            lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
            upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

            if i>upper_bound:
                LB_sum=LB_sum+(i-upper_bound)**2
            elif i<lower_bound:
                LB_sum=LB_sum+(i-lower_bound)**2

        return np.sqrt(LB_sum)

