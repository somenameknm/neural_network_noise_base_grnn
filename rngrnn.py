import math
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class RNGRNN(BaseEstimator, ClassifierMixin):
    """A General Regression Neural Network extended method.
    Parameters:
    ----------        
    sigma : float, default=0.1
        Bandwidth standard deviation parameter for the GRNN.
    
    D : int, default=10
        Amount of additional vectors pairs

    dev : float, default=1
        Standard deviation parameter for generating noise
    """
    def __init__(self, sigma=0.1, D=10, dev=1):
        self.sigma = 2 * np.power(sigma, 2)
        self.D = D
        self.dev = dev

    def fit(self, X, y):
        """Fit the model.  
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples. Generally corresponds to the training features
        y : array-like, shape = [n_samples]
            The output or target values. Generally corresponds to the training targets
        Returns
        -------
        self : object
            Returns self.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.noise = np.random.rand(self.D, X.shape[1]) * self.dev

        self.is_fitted_ = True
        # Return the regressor
        return self

    def predict(self, X):
        """Predict target values for X.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples. Generally corresponds to the testing features
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted target value.
        """

        # Check if fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        def _grnn(X):
            EPS = math.pow(10, -7)        
            gausian_distances = np.exp(-np.power(np.sqrt((np.square(self.X_ - X).sum(axis=1))), 2) / self.sigma)
            gausian_distances_sum = gausian_distances.sum()
            if gausian_distances_sum < EPS:
                gausian_distances_sum = EPS

            return np.multiply(gausian_distances, self.y_).sum() / gausian_distances_sum

        def _process(sample):
            # Create additional vectors for every test vector
            extended_sample = np.concatenate([sample - self.noise, sample + self.noise, [sample]])
            # For every augmented vector and base make prediction using grnn and average the results
            return np.average(np.apply_along_axis(_grnn, axis=1, arr=extended_sample))

        # For every test vector make augmentation, use grnn and average the results
        return np.apply_along_axis(_process, axis=1, arr=X)
