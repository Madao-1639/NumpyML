import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
# Original thought:
    # def sigmoid(x):
    #     if x>=0:
    #         return 1/(1+np.exp(-x))
    #     else:
    #         return np.exp(x)/(1+np.exp(x))
# But counter overflow! Use 'expit' instead.

class logreg:
    w = None
    b = 0
    def __init__(self,penalty=None,_lambda=1,_delta =0.3):
        self.penalty = penalty
        self._lambda = _lambda
        self._delta = _delta
    def fit(self,X,y,batch=None,lr=1e-2,iter=1000):
        if X.shape[0]==y.size: # Formulation
            X=X.T
        if self.w is None: # Initialize w
            w = np.zeros(X.shape[0])
        b = self.b
        _range = range(y.size)
        for t in range(iter):
            if batch != None: # Initialize batch
                _range = np.random.choice(range(y.size),batch) # SGD
            else:
                batch = y.size
            if self.penalty == 'l1':    # "Huber"
                g = w
                g[np.abs(w)<self._delta] = g[np.abs(w)<self._delta]/self._delta
                g[(np.abs(w)>=self._delta) & (w<0)] = -1
                g[(np.abs(w)>=self._delta) & (w>0)] = 1
                g = g*self._lambda
            elif self.penalty =='l2':
                g=2*self._lambda*w
            else:
                g=0
            gb = 0
            for i in _range:
                tmp = expit(w@X[:,i]+b) - y[i]
                gb+= tmp
                g+=tmp*X[:,i]
            w=w-(lr/batch)*g # Fix Steps
            b=b-(lr/batch)*gb
        self.w = w
        self.b = b
        return self
    def eval(self,X):
        if X.ndim>1 and X.shape[1]==self.w.size:
            X=X.T
        return expit(self.w@X + self.b)
    def predict(self,X,thres=0.5):
        return np.where(self.eval(X)>=thres,1,0)
    
if __name__=='__main__':
    from sklearn.datasets import make_classification
    X,y = make_classification(
        n_samples=100, n_classes=2,
        n_features=2, n_informative=2, n_repeated=0, n_redundant=0,
        random_state=42)

    model = logreg()
    model.fit(X,y)
    plot_x=np.arange(X[:,0].min(),X[:,1].max(),0.1)
    plot_y=-(model.b+plot_x*model.w[0])/model.w[1]

    plt.figure(figsize=(15,15))
    plt.scatter(X[y==1][:,0],X[y==1][:,1],c='k')
    plt.scatter(X[y==0][:,0],X[y==0][:,1], color='none',marker='o', edgecolors='k')
    plt.plot(plot_x,plot_y,'k')
    plt.legend(['y=1','y=0','LR'])
    plt.savefig('LR result.jpg',bbox_inches='tight')