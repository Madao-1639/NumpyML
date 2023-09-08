import numpy as np
import matplotlib.pyplot as plt

def POCKET(x,y,threshold,s=1,iteration=1000):
    n=y.size
    x=np.append(np.ones((n,1)),x,axis=1)
    w=np.zeros(x.shape[1])
    w[0]=threshold
    
    error=(np.sign(x@w)==y)
    tol_error = error.sum()
    for i in range(iteration):
        errorIdx = np.argwhere(error).reshape(-1)
        for k in np.random.permutation(errorIdx):
            w_hat = w + s*y[k]*x[k]
            error_hat = (np.sign(x@w_hat)!=y)
            tol_error_hat = error_hat.sum()
            if tol_error_hat<tol_error:
                w = w_hat
                error = error_hat
                tol_error = tol_error_hat
                break
    return w[0],w[1:]

if __name__=='__main__':
    from sklearn.datasets import make_classification
    X,y = make_classification(
        n_samples=100, n_classes=2, flip_y=0.05,
        n_features=2, n_informative=2, n_repeated=0, n_redundant=0,
        random_state=42)
    y[y==0]=-1

    threshold = 1
    threshold,w=POCKET(X,y,threshold)
    plot_x=np.arange(X[:,0].min(),X[:,1].max(),0.1)
    plot_y=-(threshold+plot_x*w[0])/w[1]

    plt.figure(figsize=(15,15))
    plt.scatter(X[y==1][:,0],X[y==1][:,1],c='k')
    plt.scatter(X[y==-1][:,0],X[y==-1][:,1], color='none',marker='o', edgecolors='k')
    plt.plot(plot_x,plot_y,'k')
    plt.legend(['y=1','y=-1','Pocket'])
    plt.savefig('Pocket result.jpg',bbox_inches='tight')