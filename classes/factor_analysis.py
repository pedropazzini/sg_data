import numpy as np

class Factor_Analysis:

    # Sample covariance matrix
    __S = None
    # Eigenvalues vector
    __w = None
    # Eigenvectors matrix
    __v = None
    # Sample values
    __data = None
    # Number of eigenvectors
    __n_eigenvectors = None
    # The loading Factors matrix
    __L = None
    # The error matrix
    __Psi = None


    def __init__(self, data, threshold = 0.95):
        
        self.__data = data
        centered_data = data - np.mean(data,axis=0)

        self.__S = np.corrcoef(centered_data)
        self.__w, self.__v = np.linalg.eig(centered_data)

        self.__L = np.dot(np.srqt(self.__w),self.__v)
        self.__L = self.__L.T

        self.__build_Psi()

        self.__get_number_of_factors()

    def __build_Psi(self):

        diag = self.__S.diagonal()
        psii = []
        for i in range(0, len(diag)):
            psii.append(diag[i] - sum(np.power(self.__L[i],2)))

        self.__Psi = np.identity(len(diag))*np.array(psii)

    def __get_number_of_factors(self):
        
        cumsum_sqd_w = np.cumsum(np.power(self.__w,2))
        residual = (self.S-(np.dot(self.__L,self.__L.T) + self.__Psi))
        sum_squared_entries = sum(np.power(residual,2))
        self.__n_eigenvectors = sum(residual <= cumsum_sqd_w)

    def get_laod_factors(self):
        return self.__L

    def get_n_eigenvectors(self):
        return self.__n_eigenvectors




