import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(r"""C:\Users\NARESH\Documents\handwritten\machine-learning-online-2018-master\5. K-Nearest Neighbours\train.csv""")

data=df.values

X=data[:,1:]
Y=data[:,0]

split=int(.8*(X.shape[0]))


X_train = X[:split,:]
Y_train = Y[:split]

X_test = X[split:,:]
Y_test = Y[split:]

def drawImg(sample):
    img = sample.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.show()
    
drawImg(X_train[3])
print(Y_train[0])


def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,queryPoint,k=12):
    
    vals = []
    m = X.shape[0]
    
    for i in range(m):
        d = dist(queryPoint,X[i])
        vals.append((d,Y[i]))
        
    
    vals = sorted(vals)
    # Nearest/First K points
    vals = vals[:k]
    
    vals = np.array(vals)
    
    #print(vals)
    
    new_vals = np.unique(vals[:,1],return_counts=True)
    #print(new_vals)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    
    return pred


pred=knn(X_train,Y_train,X_test[1])

drawImg(X_test[1])
# print(Y_train[0])
print(Y_test[1])

