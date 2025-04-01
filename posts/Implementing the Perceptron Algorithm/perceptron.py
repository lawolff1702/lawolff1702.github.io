import torch

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1])) # size p

        return torch.matmul(X, self.w) # same as X @ self.w

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """

        return 1.0 if self.score(X) > 0 else 0.0


class Perceptron(LinearModel):

    def loss(self, X, y):
        """
        Compute the misclassification rate. A point i is classified correctly if it holds that s_i*y_i_ > 0, where y_i_ is the *modified label* that has values in {-1, 1} (rather than {0, 1}). 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        """
        y_ = 2*y - 1

        return (1.0*((self.score(X)*y_) < 0)).mean()
        

    def grad(self, X, y):
        y_ = 2*y - 1
        return (-((1.0*((self.score(X) * y_) < 0))*y_).unsqueeze(1) * X).mean(dim=0) # hyperparameter of 1


class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y, k=1, learn=1.0):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """
        ix = torch.randperm(X.size(0))[:k]
        batch_X, batch_y = X[ix, :], y[ix]

        self.model.loss(X, y)
        self.model.w -= learn * self.model.grad(batch_X, batch_y) 