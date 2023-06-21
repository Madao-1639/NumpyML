from math import exp
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    if x>=0:
        return 1.0/(1+np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))

def LG(x,y, w,a=0.01,tolerence=1e-2):
    n=y.size
    yx=(y*x).T
    steps=np.floor(1/(a*tolerence)).astype(int)
    for i in range(steps):
        theta=np.apply_along_axis(lambda ynxn:sigmoid(-w.T@ynxn)*(-ynxn),axis=0,arr=yx)
        g=theta.sum(axis=1).reshape(2,1)/n
        w=w-a*g#Fix Steps
    return w


if __name__=='__main__':
    n=100
    np.random.seed(42)
    
    #以随机生成分界线，在两侧随机生成数据
    random1=np.random.random(2)#随机分界线的法向量
    random2=200*np.random.random(n)-100#分界线两侧数据的偏移量
    x1=200*np.random.random(n)-100
    x2=x1*random1[0]/random1[1]+random2

    x=np.array(list(zip(x1,x2)))
    y=np.zeros((n,1))
    y[random2>0]=1#设分界线上方的点标签为1
    y[random2<0]=-1#设分界线下方的点标签为-1
    w0=np.array([0,0]).reshape(2,1)

    w=LG(x,y,w0)
    plot_x=np.arange(-100,100,0.1)
    random_divide=plot_x*random1[0]/random1[1]
    result_divide=-1*plot_x*w[0]/w[1]

    y=y.reshape(n)
    plt.figure(figsize=(15,15))
    plt.scatter(x[y==1][:,0],x[y==1][:,1],c='k')
    plt.scatter(x[y==-1][:,0],x[y==-1][:,1], color='none',marker='o', edgecolors='k')
    plt.plot(plot_x,random_divide,'k--')
    plt.plot(plot_x,result_divide,'k')
    plt.legend(['y=1','y=-1','random divide','result divide'])
    plt.savefig('Logistic_Rgression_result.jpg',bbox_inches='tight')