import torch
from scipy.optimize import minimize

class KernelLogisticRegression:

    def __init__(self, kernel, **kwargs):
        self.kernel = kernel
        self.kwargs = kwargs

    def fit(self, X, y):
        self.X_train = X 

        K = self.kernel(X, X, **self.kwargs)

        self.a = torch.zeros(X.size(0), dtype=torch.float32) # Initialize alpha to zeros

        s = minimize(lambda a: self.BCEloss(K, y), a0 = self.a)

        self.a = s.x
                
    def predict(self, X):

        
    def score(self, X, y):
        y_hat = self.predict(X)
        return (y_hat == y).mean()
    
    def BCEloss(self, X, y):
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
        sig = torch.clamp(torch.sigmoid(s), 1e-7, 1 - 1e-7) # to avoid log(0) --> nan
        return (-y * torch.log(sig) - (1 - y) * torch.log(1 - sig)).mean()
    

from logistic import LogisticRegression, GradientDescentOptimizer

class KernelLogisticRegression:
    def __init__(self, kernel, lam=1.0, **kernel_kwargs):
        """
        Sparse Kernel Logistic Regression with L1 regularization.

        Args:
            kernel (callable): fn(X1, X2, **kernel_kwargs) → (n1 x n2) kernel matrix
            lam (float): L1 penalty strength
            **kernel_kwargs: passed on to `kernel`
        """
        self.kernel    = kernel
        self.lam       = lam
        self.kernel_kwargs = kernel_kwargs

    def fit(self, X, y, epochs=1000, lr=1e-2, verbose=False):
        """
        Fit by gradient-descent on o.
        
        Args:
            X (Tensor[m,d]): training inputs
            y (Tensor[m]):   binary labels (0 or 1)
            epochs (int):    number of GD steps
            lr (float):      learning rate
        """
        self.X_train = X
        m = X.size(0)

        # Precompute Gram matrix once
        K = self.kernel(X, X, **self.kernel_kwargs)  # (m x m)

        # α as a parameter so autograd can compute ∂/∂α
        self.a = torch.zeros(m, requires_grad=True)

        LR = LogisticRegression() 
        opt = GradientDescentOptimizer(LR)

        for t in range(1, epochs+1):
            s = K.matmul(self.a) # raw scores
            # BCE with logits is numerically stable
            loss_bce = LR.loss(s, y)
            loss_l1  = self.lam * torch.norm(self.a, p=1) / m
            loss = loss_bce + loss_l1
            loss.backward()
            opt.step(s, y, alpha=0.01, beta=0.9)

            if verbose and t % (epochs//10 or 1) == 0:
                nnz = (self.a.abs() > 1e-6).sum().item()
                print(f"Epoch {t:4d}/{epochs}  loss={loss.item():.4f}  nnz={nnz}")

        # after fit, no more grads needed
        self.a = self.a.detach()

    def score(self, X, recompute_kernel=False):
        """
        Raw scores s = K(X, X_train) @ a
        """
        if recompute_kernel or X is not self.X_train:
            Kx = self.kernel(X, self.X_train, **self.kernel_kwargs)
        else:
            Kx = None  # not used
        Kx = Kx if Kx is not None else self.kernel(X, self.X_train, **self.kernel_kwargs)
        return Kx.matmul(self.a)

    def predict(self, X, threshold=0.5):
        """
        Binary predictions: sigmoid(score) > threshold
        """
        probs = torch.sigmoid(self.score(X, recompute_kernel=True))
        return (probs > threshold).float()

    def accuracy(self, X, y):
        """
        Fraction of correct predictions on (X,y)
        """
        return (self.predict(X) == y).float().mean()