import pandas as pd
from math import log
#import sys
#sys.setrecursionlimit(3000)

class DecisionTree:
    '''
    If it's a leaf, it has a label.
    If it's not, it has a test and a next (node dict).
    '''
    isLeaf=False
    def __init__(self,depth=1,method='id3'):
        self.depth=depth
        if method in ['id3','c4.5','cart']:
            self.method=method
        else:
            raise BaseException('Wrong method!')

    def Entropy(self,column):
        '''
        Calculate entropy of a list.
        '''
        Ent=0
        n=len(column)
        for x in set(column):
            count=0
            for val in column:
                if x==val:
                    count+=1
            p=count/n
            Ent=Ent-p*log(p)
        return Ent

    def Gini(self,column):
        '''
        Calculate Gini of a list.
        '''
        gini=1
        n=len(column)
        for x in set(column):
            count=0
            for val in column:
                if x==val:
                    count+=1
            p=count/n
            gini=gini-p*p
        return gini


    def selectAttr(self,D,A):
        '''
        Select attr with best score.
        if:
            method is id3, score is Gain
            method is c4.5, score is Gain_Ratio
            method is cart, score is Gain_Gini
        '''
        scores={attr:0 for attr in A}
        n=D.shape[0]
        #Calculate score for each attr
        if self.method=='id3' or self.method=='c4.5':
            H_D=self.Entropy(D.iloc[:,-1])
            for attr in A:
                H_DA=0
                for x in set(D[attr]):
                    p=D[D[attr]==x].shape[0]/n
                    H_DA=H_DA+p*self.Entropy(D[D[attr]==x].iloc[-1])
                Gain=H_D-H_DA
                score=Gain
                if self.method=='c4.5':
                    H_A=self.Entropy(D[attr])
                    Gain_Ratio=Gain/H_A
                    score=Gain_Ratio
                scores[attr]=score
        else:
            Gini_D=self.Gini(D.iloc[:,-1])
            for attr in A:
                Gini_DA=0
                for x in set(D[attr]):
                    p=D[D[attr]==x].shape[0]/n
                    Gini_DA=Gini_DA+p*self.Gini(D[D[attr]==x].iloc[-1])
                Gain_Gini=Gini_D-Gini_DA
                score=Gain_Gini
                scores[attr]=score
    
        #Select the best attr
        MaxScore=0
        for attr in scores.keys():
            if scores[attr]>MaxScore:
                MaxScore=scores[attr]
                BestAttr=attr
        return BestAttr

    def fit(self,D,A=None,max_depth=None):
        """
        --------------------------------------
        Input Format:
        D=                   
        [[x11,x12,...,x1n,y1],
         [x21,x22,...,x2n,y2],
        ...                     (2-D array contains tests)
         [xm1,xm2,...,xmn,ym]]

        A=['attr1','attr2',...,'attrm']
        --------------------------------------
        """

        #make sure inputed D is processed as a DataFrame.
        if type(D)==pd.core.frame.DataFrame:
            columns[:-1]='y'
            if A==None:
                A=D.columns[:-1]
        else:
            columns=A+'y'
            D=pd.DataFrame(D,columns=columns)
        
        y=D.iloc[:,-1]
        if (y==y.iloc[0]).all():#All tests are the same, dont need to devide
            self.label=y.iloc[0]
            self.isLeaf=True
        elif D.shape[1]==1 or (D.iloc[:,:-1]==D.iloc[0,:-1]).all().all() or (max_depth!=None and self.depth==max_depth):#lost attribute or values are the same or reach the max_depth, cant be devided
                    self.label=y.value_counts().argmax()
                    self.isLeaf=True
        else:
            attr=self.SelectAttr(D,A)
            self.test=attr
            self.next=dict()
            for val in set(D[attr]):
                next_D=D[D[attr]==val]
                next_Tree=DecisionTree(depth=self.depth+1,method=self.method)
                if next_D.shape[0]==0:
                    next_Tree.label=y.value_counts().argmax()
                    next_Tree.isLeaf=True
                else:
                    next_Tree.fit(next_D,max_depth=max_depth)
                self.next[val]=next_Tree
        return self

    def deside(self,x):
        '''
        Send a single dict-type instance X that contains attrs and vals.
        '''
        if self.isLeaf:
            return self.label
        else:
            return self.next[x[self.test]].deside(x)