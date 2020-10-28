import numpy as np
class AdalineSGD(object):
    """Adaline Classifier.
    
    Trains using stochastic gradient descent. Has a few more functions than Batch GD.

    Parameters
    ----------
    eta: float
        Learning rate
    n_iter: int
        Number of epochs
    shuffle: bool (default: True)
        Shuffles training data every epoch to prevent cycles
    random_state: int
        Seed for RNG
    
    Atrributes
    ----------
    w_: 1d array
        Weights vector, updates while training
    cost_: list
        Keeps track of cost function value in each epoch
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        # eta and n_iter are hyperparameters
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
    
    def fit(self, X, y):
        """Trains Adaline with Widrow-Hoff method (using Stochastic Gradient Descent).

        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Input matrix with n_samples rows and n_features columns
        y: array-like, shape = [n_samples]
            Input column vector with n_samples rows and the correctly labeled outputs
        
        Returns
        -------
        self: object
        """

        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fits training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
    
    def _shuffle(self, X, y):
        """Shuffles training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initializes weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """Calculates input but taking dot product of X with w_"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Computes linear activation"""
        return X
    
    def predict(self, X):
        """Returns class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)