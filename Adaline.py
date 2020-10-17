import numpy as np
class Adaline(object):
    """Adaline Classifier.

    Parameters
    ----------
    eta: float
        Learning rate
    n_iter: int
        Number of epochs
    random_state: int
        Seed for RNG
    
    Atrributes
    ----------
    w_: 1d array
        Weights vector, updates while training
    cost_: list
        Keeps track of cost function value in each epoch
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """Trains Adaline with Widrow-Hoff method.

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

        #initializes the weight vector with random, nonzero values
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        
        self.cost_ = []

        for i in range(self.n_iter):

            #calls function that performs dot product for X and w_
            net_input = self.net_input(X)

            #calculates the output needed for training (using linear activation)
            output = self.activation(net_input)

            #calculates error array for updating w_
            errors = (y - output)

            #updates all weight values (with Widrow-Hoff) except bias (w_[0])
            self.w_[1:] += self.eta * X.T.dot(errors)

            #updates bias value
            self.w_[0] += self.eta * errors.sum()

            #calculates value of cost function for this epoch, then appends to cost_ list
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

        def net_input(self, X):
            """Calculates input but taking dot product of X with w_"""
            return np.dot(X, self.w_[1:]) + self.w_[0]

        def activation(self, X):
            """Computes linear activation"""
            return X
        
        def predict(self, X):
            """Returns class label after unit step"""
            return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
        
        print("Hello")