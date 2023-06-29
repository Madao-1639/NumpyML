import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    if x>=0:
        return 1/(1+np.exp(-x))
    else:
        return np.exp(x)/(1+np.exp(x))

class logreg:
    def __init__(self,batch=None,lr=1e-2) -> None:
        self.batch = batch
        self.lr=lr
    def fit(self,X,y,iter=1000):
        if X.shape[0]==y.size:
            X=X.T
        w = np.zeros(X.shape[0])
        _range = range(y.size)
        for i in range(iter):
            if self.batch != None:
                _range = np.random.choice(range(y.size),self.batch)
            g=0
            for i in _range:
                g+=(y[i]*sigmoid(-w@X[:,i])+(1-y[i])*sigmoid(w@X[:,i]))*X[:,i]
            w=w-self.lr*g#Fix Steps
        self.w = w
        return w
    def eval(self,X):
        if X.shape[0]==self.w.size:
            X=X.T
        tmp = self.w@X.T
        tmp = tmp.reshape(-1,1)
        tmp = np.apply_along_axis(sigmoid,1,tmp)
        tmp = tmp.ravel()
        return tmp
    def predict(self,X,thres=0.5):
        return np.where(self.eval(X)>=thres,1,0)
if __name__=='__main__':
    import matplotlib.pyplot as plt
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
    y[random2<0]=0#设分界线下方的点标签为0

    model = logreg()
    w=model.fit(x,y)
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