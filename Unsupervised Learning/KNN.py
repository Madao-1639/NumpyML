import pandas as pd
import numpy as np

def Euc_distance(a,b):
    return np.linalg.norm(a.values-b.values)

def Mht_distance(a,b):
    return np.abs(a.values-b.values).sum()

def Chbyshv_distance(a,b):
    return np.abs(a.values-b.values).max()

class KDitem:
    def __init__(self,node,dist,next=None):
        self.node=node
        self.dist=dist
        self.next=next

class KDstack:
    def __init__(self,k=1):
        self.n=0
        self.top=None
        self.k=k
    def push(self,node,dist):
        _item1=self.top=KDitem(node,dist,self.top)
        upper=None
        self.n+=1
        while _item1.next!=None and _item1.dist<_item1.next.dist:
            _item2=_item1.next
            _item1.next=_item2.next
            _item2.next=_item1
            if upper==None:
                upper=self.top=_item2
            else:
                upper.next=_item2
                upper=upper.next
        if self.n>k:
            self.top=self.top.next
    def pop(self):
        if self.top!=None:
            item=self.top
            self.top=self.top.next
            return item
        else:
            return None
    def exsist(self,node):
        temp=self.top
        while temp!=None:
            if temp.node==node:
                return True
        return False
    def to_frame(self):
        temp=self.top
        p_list=[]
        while temp!=None:
            p_list.append(temp.node.thresP)
            temp=temp.next
        return pd.concat(p_list)

stack=KDstack()

class KDtree:
    def __init__(self):
        pass
    
    def build(self,data):
        if data.size==0:
            return None
        elif data.shape[0]==1:
            self.thresP=data
            self.lt=self.rt=None
            return self
        else:
            self.dimension=data.std().idxmax()
            self.thresP=data.sort_values(self.dimension).iloc[[data.shape[0]//2]]
            data=data.drop(self.thresP.index)
            self.lt=KDtree().build(data[data[self.dimension]<=self.thresP[self.dimension].iloc[0]])# left tree
            self.rt=KDtree().build(data[data[self.dimension]>self.thresP[self.dimension].iloc[0]])# right tree
            return self

    def search(self,x):
        if self.lt==self.rt==None:
            return [self]
        elif x[self.dimension]<=self.thresP[self.dimension].iloc[0]:
            if self.lt!=None:
                next_search=self.lt.search(x)
                next_search.insert(0,self)
                return next_search
            else:
                return [self]
        else:
            if self.rt!=None:
                next_search=self.rt.search(x)
                next_search.insert(0,self)
                return next_search
            else:
                return [self]

    def neighborhood(self,x,method):
        global stack
        path=self.search(x)
        node=path.pop()# near: neareast node, node: current node
        dist=method(node.thresP,x)
        stack.push(node,dist)
        while len(path)>0:
            parent_node=path.pop()
            parent_p=parent_node.thresP
            parent_dist=method(parent_p,x)
            stack.push(parent_node,parent_dist)
            if parent_p[parent_node.dimension].iloc[0]>x[parent_node.dimension]-stack.top.dist and \
                parent_p[parent_node.dimension].iloc[0]<x[parent_node.dimension]+stack.top.dist:
                if node==parent_node.lt and parent_node.rt!=None:
                    parent_node.rt.neighborhood(x,method)
                elif node==parent_node.rt and parent_node.lt!=None:
                    parent_node.lt.neighborhood(x,method)
            node=parent_node
            
    
    def knn(self,k,x,method):    
        global stack
        stack.k=k
        self.neighborhood(x,method)
        neighborhood=stack.to_frame()
        return neighborhood
                
class KNN:
    def fit(self,data,labels,algorithm='auto',distance='Euclidean'):
        if distance=='Euclidean':
            self.method=Euc_distance
        elif distance=='Manhattan':
            self.method=Mht_distance
        else:
            self.method=Chbyshv_distance

        if algorithm=='auto':
            if data.shape[0]>=500:
                if data.shape[1]>=20:
                    algorithm='ball_tree'
                else:
                    algorithm='kd_tree'
            else:
                algorithm='brute'

        self.labels=labels
        if algorithm=='brute':
            self.tree=None
            self.data=data
        elif algorithm=='kd_tree':
            self.tree=KDtree().build(data)
        else:
            pass
            #self.tree=
    
    def predict(self,k,X):
        if len(X.shape)==1:# X is a Series
            X=X.to_frame().T
        self.y_pred=np.empty(X.shape[0])
        if self.tree==None:
            for i in range(X.shape[0]):
                x=X.iloc[i]
                allDist=self.data.apply(lambda p:self.method(p,x),\
                    axis=1).sort_values(ascending=True)
                self.neighborhood=allDist.head(k)
                labels=self.labels[self.neighborhood.index]
                self.y_pred[i]=labels.max()
        else:
            for i in range(X.shape[0]):
                x=X.iloc[i]
                self.neighborhood=self.tree.knn(k,x,self.method)
                labels=self.labels[self.neighborhood.index]
                self.y_pred[i]=labels.max()
        return self.y_pred



if __name__=='__main__':
    import matplotlib.pyplot as plt
    data=pd.DataFrame({0:[1,3,5,7,9,2,4,6,8,2],1:[10,38,57,28,40,7,1,8,15,14]})
    target=pd.Series([1,2,3,2,1,1,2,1,3,2])
    X=pd.Series([1.5,10])
    k=2

    model=KNN()
    model.fit(data,target,algorithm='brute')
    y_pred=model.predict(k,X)
    kd_model=KNN()
    kd_model.fit(data,target,algorithm='kd_tree')
    y_kd_pred=kd_model.predict(k,X)

    plt.figure(figsize=(20,10))
    plt.subplot(121)
    plt.axis('equal')
    plt.title('brute')
    plt.scatter(data[target==1][0],data[target==1][1],c='r')
    plt.scatter(data[target==2][0],data[target==2][1],c='g')
    plt.scatter(data[target==3][0],data[target==3][1],c='b')
    plt.scatter(X[0],X[1],c='k')
    neighborhood=data.iloc[model.neighborhood.index]
    for i in range(neighborhood.shape[0]):
        neighbor=neighborhood.iloc[i]
        concate=pd.concat((X,neighbor)).T
        plt.plot(concate[0],concate[1])

    plt.subplot(122)
    plt.axis('equal')
    plt.title('kd_tree')
    plt.scatter(data[target==1][0],data[target==1][1],c='r')
    plt.scatter(data[target==2][0],data[target==2][1],c='g')
    plt.scatter(data[target==3][0],data[target==3][1],c='b')
    plt.scatter(X[0],X[1],c='k')
    kd_neighborhood=data.iloc[kd_model.neighborhood.index]
    for i in range(kd_neighborhood.shape[0]):
        kd_neighbor=kd_neighborhood.iloc[i]
        concate=pd.concat((X,kd_neighbor)).T
        plt.plot(concate[0],concate[1])

    plt.show()