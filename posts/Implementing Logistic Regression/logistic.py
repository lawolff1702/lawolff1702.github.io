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

class LogisticRegression(LinearModel):
    """
    Logistic regression model. Inherits from LinearModel.
    """

    def loss(self, X, y):
        """
        Compute the binary cross-entropy loss / empirical risk. 
        The loss for a single data point i is given by the formula:
        L_i = -y_i*log(sig(s_i)) - (1 - y_i)*log(1 - sig(s_i)), where s_i is the score for data point i and sig(s) is 1 / (1 + e^(-s)).

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are between (0, 1)
        RETURNS: 
            L, torch.Tensor: mean loss for the data points
        """
        s = self.score(X)
        sig = torch.sigmoid(s)
        return (-y * torch.log(sig) - (1 - y) * torch.log(1 - sig)).mean()
    
    def grad(self, X, y):
        """
        Compute the gradient of the binary cross-entropy loss / empirical risk.
        The gradient for a single data point i is given by the formula:
        g_i = (sig(s_i) - y_i)*x_i, where s_i is the score for data point i and sig(s) is 1 / (1 + e^(-s)).
        
        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are between (0, 1)
        RETURNS: 
            g, torch.Tensor: the gradient of the loss for the data point. g.size() = (n, p)   
        """
        s = self.score(X)
        return ((torch.sigmoid(s) - y)[:,None] * X).mean(dim=0)


class GradientDescentOptimizer:
    """
    Gradient descent optimizer for the logistic regression model.
    """

    def __init__(self, model):
        self.model = model
        self.w_prev = None 
    
    def step(self, X, y, alpha=0.1, beta=0.9):
        """
        Perform a single step of gradient descent. 
        The formula for the update is:
        w <- w - alpha * g + beta * (w - w_prev),
        where g is the gradient of the loss function and w_prev is the previous value of w.
        alpha is the learning rate and beta is the momentum term.

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are between (0, 1)

            alpha, float: learning rate

            beta, float: momentum term
        RETURNS:
            None: updates the self.w in place
        """
        self.model.loss(X, y)

        if self.w_prev is None: 
            self.w_prev = self.model.w.clone()
            
        w_new = self.model.w - (alpha * self.model.grad(X, y)) + (beta * (self.model.w - self.w_prev))

        self.w_prev = self.model.w.clone()
        self.model.w = w_new
