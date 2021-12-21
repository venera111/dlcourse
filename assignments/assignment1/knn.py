import numpy as np
from scipy.stats import mode
from scipy.spatial.distance import cdist

class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0] # number of training data
        num_test = X.shape[0]             # number of test data
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                dists[i_test, i_train] = np.sum(np.abs(X[i_test] - self.train_X[i_train]))
#                 dists[i_test, i_train] = np.sqrt(np.sum(np.square(X[i_test,:] - self.train_X[i_train,:])))    L2
        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            dists[i_test] = np.sum(np.abs(X[i_test] - self.train_X), axis=1)
#             dists[i_test] = np.sqrt(np.sum(np.square(self.train_X - X[i_test,:]), axis = 1))   L2
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        dists = np.sum(np.abs(X[:, None, :] - self.train_X[None, :, :]), axis=-1)
#         dists = np.sqrt((X**2).sum(axis=1)[:, np.newaxis] + (self.train_X**2).sum(axis=1) - 2 * X.dot(self.train_X.T)) L2
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, bool)
        for i in range(num_test):
            closest_y = np.take(self.train_y, np.argsort(dists[i]))[:self.k]
            (values, counts) = np.unique(closest_y, return_counts=True)
            pred[i] = values[np.argmax(counts)]
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        class_dict = {}
        pred = np.zeros(num_test, np.int)

        for cls in self.train_y:
            if not cls in class_dict:
                class_dict[cls] = 0
        
        for i in range(num_test):
            class_dict = dict.fromkeys(class_dict.keys(), 0)
            index = dists[i, :].argsort()[:self.k]
            for idx in index:
                class_dict[self.train_y[idx]] += 1
            pred[i] = max(class_dict, key=lambda key: class_dict[key])
        return pred