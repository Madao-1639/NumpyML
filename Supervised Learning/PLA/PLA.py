import numpy as np
import matplotlib.pyplot as plt

def PLA(x,y,threshold,s=1,iteration=1000):
    n=y.size
    x=np.append(np.ones((n,1)),x,axis=1)
    w=np.zeros(x.shape[1])
    w[0]=threshold
    
    for i in range(iteration):
        error = (np.sign(x@w)!=y)
        if error.sum() == 0:
            break
        errorIdx = np.argwhere(error).reshape(-1)
        j = np.random.choice(errorIdx)
        w=w + s*y[j]*x[j]
    return w[0],w[1:]

if __name__=='__main__':
    from sklearn.datasets import make_classification
    X,y = make_classification(
        n_samples=100, n_classes=2,
        n_features=2, n_informative=2, n_repeated=0, n_redundant=0,
        random_state=42)
    y[y==0]=-1

    threshold = 1
    threshold,w=PLA(X,y,threshold)
    plot_x=np.arange(X[:,0].min(),X[:,1].max(),0.1)
    plot_y=-(threshold+plot_x*w[0])/w[1]

    plt.figure(figsize=(15,15))
    plt.scatter(X[y==1][:,0],X[y==1][:,1],c='k')
    plt.scatter(X[y==-1][:,0],X[y==-1][:,1], color='none',marker='o', edgecolors='k')
    plt.plot(plot_x,plot_y,'k')
    plt.legend(['y=1','y=-1','PLA'])
    plt.savefig('PLA result.jpg',bbox_inches='tight')