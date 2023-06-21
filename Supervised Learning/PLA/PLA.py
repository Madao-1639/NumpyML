import numpy as np
import matplotlib.pyplot as plt

def sign(x):
    if x>0:
        return 1
    elif x<0:
        return -1
    else:
        return 0

def PLA(x,y,threhold, w=np.array([0,0])):
    n=y.size
    x=np.append(np.ones((n,1)),x,axis=1)
    w=np.insert(w,0,-1*threhold)
    
    error=np.ones(n)
    i=0
    while error.sum()>0:
        if sign(w@x[i].T)==y[i]:
            error[i]=0
        else:
            w=w+y[i]*x[i]
            error[:]=1
        i=(i+1)%n
    return w[1:]

if __name__=='__main__':
    n=100
    threhold=1
    np.random.seed(42)
    
    #以随机生成分界线，在两侧随机生成数据
    random1=np.random.random(2)#随机分界线的法向量
    random2=200*np.random.random(n)-100#分界线两侧数据的偏移量
    x1=200*np.random.random(n)-100
    x2=x1*random1[0]/random1[1]+random2
    x=np.array(list(zip(x1,x2)))
    y=np.zeros(n)
    y[random2>0]=1#设分界线上方的点标签为1
    y[random2<0]=-1#设分界线下方的点标签为-1

    w=PLA(x,y,threhold)
    plot_x=np.arange(-100,100,0.01)
    random_divide=plot_x*random1[0]/random1[1]
    result_divide=-1*plot_x*w[0]/w[1]

    plt.figure(figsize=(15,15))
    plt.scatter(x[y==1][:,0],x[y==1][:,1],c='k')
    plt.scatter(x[y==-1][:,0],x[y==-1][:,1], color='none',marker='o', edgecolors='k')
    plt.plot(plot_x,random_divide,'k--')
    plt.plot(plot_x,result_divide,'k')
    plt.legend(['y=1','y=-1','random divide','result divide'])
    plt.savefig('PLA_result.jpg',bbox_inches='tight')