from scipy.stats import pearsonr
import numpy as np

# Based on the paper http://eprints.uni-mysore.ac.in/3921/1/10.1007_s11634-006-0004-6

class Correlation_based_distance:

    def __init__(self,s1,s2,type_=1,beta=1.0):

        if(not (type_ == 1 or type_ == 2)):
            raise ValueError("Only '1' and '2' available for type_...")

        if (beta <=0):
            raise ValueError("Incorrect value for beta, ony positive values allowed...")

        self.beta = beta
        self.type_ = type_

        self.s1 = s1
        self.s2 = s2

    def _get_type_1(self):
        try:
            val = np.sqrt(2*(1-pearsonr(self.s1,self.s2)[0]))
            return val 
        except:
            print(self.s1)
            print(self.s2)
            print(pearsonr(self.s1,self.s2))
            raise ValueError("Erro")

    def _get_type_2(self):
        cor = pearsonr(self.s1,self.s2)[0]
        return np.sqrt(np.power((1-cor)/(1+cor),self.beta))

    def get_distance(self):
        return self._get_type_1() if self.type_== 1 else self._get_type_2()
