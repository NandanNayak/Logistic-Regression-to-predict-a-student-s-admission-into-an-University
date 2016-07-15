"""
Aim : To implement Logistic Regression
Author : Nandan Nayak
Date : 18/June/2016
"""
#import all modules
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

#define all global variables
data1="ex2data1.txt"

#define all functions
def sigmoid(val):
    return 1/(1+np.exp(-val))

def costFunc(hyp,y,m):
    result=np.empty(m)
    for i in range(m):
        if y[i]!=hyp[i]:
            result[i]=10000
        elif y[i]:
            result[i]=np.log(hyp[i])
        else:
            result[i]=np.log(1-hyp[i])
    return(-1/m*result.sum())
    


#define main function
if __name__=="__main__":
    doc1=open(data1)
    subplots=2
    x=[]
    y=[]
    
    """Reading the input"""
    for line in doc1:
        line=line.replace("\n","").split(",")
        x.append(1)
        x.append(float(line[0]))
        x.append(float(line[1]))
        y.append(int(line[2]))

    """Converting the lists to arrays"""
    x=np.array(x)
    y=np.array(y)
    m=y.shape[0]
    x=x.reshape(m,3)

    """Normalizing the input features"""
    x[:,1]=(x[:,1]-x[:,1].mean())/x[:,1].std()
    x[:,2]=(x[:,2]-x[:,2].mean())/x[:,2].std()

    """Creating empty arrays"""
    theta=np.zeros(3)
    hyp=np.empty(m)
    z=np.empty(m)
    diff=np.empty((3,m))

    """Calculating Gradient Descent"""
    alpha=0.1
    loops=1500
    J=np.empty(loops)
    for j in range(loops):
        for i in range(m):            
            z[i]=np.dot(theta.T,x[i])            
            hyp[i]=sigmoid(z[i])
            if hyp[i]>=0.5:
                hyp[i]=1
            else:
                hyp[i]=0        
        
        diff[0]=hyp-y
        diff[1]=diff[0]*x[:,1]
        diff[2]=diff[0]*x[:,2]

        for k in range(3):
            theta[k]=theta[k]-(alpha/m*diff[k].sum())

        J[j]=costFunc(hyp,y,m)

    """Calculating the accuracy of the algorithm"""
    count=0
    for i in range(m):
        if hyp[i]==y[i]:
            count+=1
    percent=count/m*100

    print "Model Parameters : ", str(theta)
    print "Percentage of examples classified correctly : %.2f"%(percent)

    """Plotting the graphs"""
    plt.subplot(subplots,1,1)
    for i in range(m):
        if y[i]:
            plt.scatter(x[i][1],x[i][2],color="g")
        else:
            plt.scatter(x[i][1],x[i][2],color="r")

    min_val= int(round(x.min()))
    max_val= int(round(x.max()))
    xL=np.arange(min_val,max_val+1)
    yL=(-theta[2]/theta[1]*xL)-(theta[0]/theta[1])
    plt.plot(xL,yL,color="blue",label="Decision Boundry")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.title("Scatter plot of training data")
    plt.text(max_val+0.5,0,"Green:Admitted",bbox=dict(boxstyle="round",color="pink"),fontsize=8)
    plt.text(max_val+0.5,-1,"Red:Rejected",fontsize=8,bbox=dict(boxstyle="round",color="pink"))
    plt.legend()

    plt.subplot(subplots,1,2)
    plt.plot(range(loops),J,color="g")
    plt.ylabel("Cost function")
    plt.xlabel("No. of iterations")
    plt.title("Variation of Cost function with each iteration")
    plt.show()
    
        
    
        
        
    

    
    
        
   
   
   
   
   
