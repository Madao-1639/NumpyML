import numpy as np
import matplotlib.pyplot as plt

def sign(x):
    if x>0:
        return 1
    elif x<0:
        return -1
    else:
        return 0

def POCKET(x,y,threhold,iter=1000,w=np.array([0,0])):
    n=y.size
    x=np.append(np.ones((n,1)),x,axis=1)
    w_hat=w=np.insert(w,0,-1*threhold)
    
    error=np.ones(n)
    error_hat=n
    for i in range(iter):
        k=n-1
        for j in range(n):
            if sign(w@x[n-1-j].T)!=y[n-1-j]:
                error[n-1-j]=1
                k=n-1-j#Backward Search to find 1st error
        error_sum=error.sum()
        if error_sum<error_hat:
            error_hat=error_sum
            w_hat=w
        if error_sum>0:
            w=w+y[k]*x[k]
            error[:]=0
        else:
            break
    return w[1:]

if __name__=='__main__':
    n=100
    threhold=1
    np.random.seed(42)
    
    x=200*np.random.random((n,2))-100
    y=np.random.randint(0,2,n)
    y[y==0]=-1

    w=POCKET(x,y,threhold)
    plot_x=np.arange(-100,100,0.01)
    result_divide=-1*plot_x*w[0]/w[1]

    plt.figure(figsize=(15,15))
    plt.scatter(x[y==1][:,0],x[y==1][:,1],c='k')
    plt.scatter(x[y==-1][:,0],x[y==-1][:,1], color='none',marker='o', edgecolors='k')
    plt.plot(plot_x,result_divide,'k')
    plt.legend(['y=1','y=-1','result divide'])
    plt.savefig('Pocket result on random D.jpg',bbox_inches='tight')