import numpy as np
import pandas as pd
import random
import math

df = pd.read_excel("Folds5x2_pp.xlsx", sheet_name="Sheet1")
df.dropna(axis=0, inplace=True)

Xtest = [[], [], [], []]
Ytest = [[]]
Xtest[0] = df.iloc[:957, 0]
Xtest[1] = df.iloc[:957, 1]
Xtest[2] = df.iloc[:957, 2]
Xtest[3] = df.iloc[:957, 3]
Ytest[0] = df.iloc[:957, 4]
Ytest = np.array(Ytest)

Xtrain = []
Ytrain = []
Xvalid = []
Yvalid = []
for i in range(len(df)-957):
    if (i+1) % 5 == 0:
        Xvalid.append(df.iloc[i+957, :4])
        Yvalid.append([df.iloc[i+957, 4]])
    else:
        Xtrain.append(df.iloc[i + 957, :4])
        Ytrain.append([df.iloc[i + 957, 4]])
Xtrain = np.transpose(Xtrain)
Ytrain = np.transpose(Ytrain)
Xvalid = np.transpose(Xvalid)
Yvalid = np.transpose(Yvalid)


def normalize(x, k):
    xnorma = []
    l = -1
    for row in x:
        xmin = min(row)
        xmax = max(row)
        xnorma.append([])
        l += 1
        for ele in row:
            xnorma[l].append((2*ele-xmin-xmax)*k/(xmax-xmin))
    return xnorma


xtrain = normalize(Xtrain, 1)
ytrain = normalize(Ytrain, 0.9)
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)
xvalid = normalize(Xvalid, 1)
xvalid = np.array(xvalid)
yvalid = normalize(Yvalid, 0.9)
xtest = normalize(Xtest, 1)
xtest = np.array(xtest)
ytest = normalize(Ytest, 0.9)


def activate(x):
    return math.tanh(x)
phi = np.vectorize(activate)

#hidden layer
w = [0, 0]
w[0] = np.random.rand(3, 4)*1/2
w[1] = np.random.rand(1, 3)*1/math.sqrt(3)

epochs = 1000
bsize = 64
batcharr = list(range(len(xtrain[0])//bsize))
yfull = []
vfull = []
eta = 0.001
hlayers = 1
validerror = []
trainingerror = []
for epoch in range(epochs):
    print(f"epoch no {epoch+1}")
    np.random.shuffle(batcharr)
    for batch in batcharr:
        y = xtrain[:, (batch*bsize): (batch+1)*bsize+1]
        ybav = np.resize(y.mean(1), (1, len(y)))
        yfull.append(ybav)
        for layer in range(2):
            v = np.dot(w[layer], y)
            y = phi(v)
            vfull.append(v.mean(1))
            ybav = np.resize(y.mean(1), (1, len(y)))
            yfull.append(ybav)
            #print(f"**layer {layer} **")
        e = ytrain[:, (batch*bsize):(batch+1)*bsize+1] - y
        ebav = e.mean(1)


    # #backprop
        deltaw = [0]*len(w)
        beta = 0.9
        lamda = 0.1
        localgrad = []
        newlocalgrad = []
        for layer in range(hlayers, -1, -1):
            if layer == hlayers:
                for j in range(len(vfull[layer])):
                    localgrad.append(ebav[j]*(1 - pow(math.tanh(vfull[layer][j]), 2)))

                deltaw[layer] = beta*deltaw[layer] + (1-beta)*eta*np.subtract(np.dot(np.resize(localgrad, (len(localgrad), 1)), yfull[layer]), lamda*np.absolute(w[layer]))
                w[layer] += deltaw[layer]
            else:
                if layer == hlayers-1:
                    localgradweight = np.dot(localgrad, w[layer+1])
                else:
                    localgradweight = np.dot(newlocalgrad, w[layer+1])
                newlocalgrad = []
                for j in range(len(vfull[layer])):
                    newlocalgrad.append((1-pow(math.tanh(vfull[layer][j]), 2))*localgradweight[j])
                deltaw[layer] = beta*deltaw[layer] + (1-beta)*eta*np.subtract(np.dot(np.resize(newlocalgrad, (len(newlocalgrad), 1)), yfull[layer]), lamda*np.absolute(w[layer]))
                w[layer] += deltaw[layer]

    #MAPEvalidation

    yvalidoutput = xvalid
    for layer in range(2):
        v = np.dot(w[layer], yvalidoutput)
        yvalidoutput = phi(v)
    evalid = yvalid - yvalidoutput
    evalid = np.divide(np.absolute(evalid), np.add(np.absolute(yvalid), 0.001))
    evalidbav = evalid.mean(1)
    error = np.sum(evalid)*100/len(yvalidoutput[0])
    validerror.append(error)


    # MAPE of training
    ytrainoutput = xtrain
    for layer in range(2):
        v = np.dot(w[layer], ytrainoutput)
        ytrainoutput = phi(v)
    etrain = ytrain - ytrainoutput
    etrain = np.divide(np.absolute(etrain), np.add(np.absolute(ytrain), 0.001))
    error = np.sum(etrain)*100/len(ytrain[0])
    trainingerror.append(error)

#MAPEtest:


ytestoutput = xtest
for layer in range(2):
    v = np.dot(w[layer], ytestoutput)
    ytestoutput = phi(v)
etest = ytest - ytestoutput
etest = np.divide(np.absolute(etest), np.add(np.absolute(ytest), 0.001))
error = np.sum(etest) * 100 / len(ytestoutput[0])
print(f"MAPEtest = {error}")
print(f"MAPEvalid = {validerror[-1]}")


#Plots


from matplotlib import pyplot as plt
plt.plot(xtrain[1], ytrainoutput[0], "y.", label="Training Output")
plt.plot(xvalid[1], yvalidoutput[0], "r.", label="Validation Output")
#plt.plot(xtrain[1], ytrain[0], ".", label="Actual Output")

plt.title("X-Y Curves After Training")
#plt.xlabel("AT")
plt.xlabel("V")
#plt.xlabel("AP")
#plt.xlabel("RH")
plt.ylabel("PE")

plt.legend(loc="upper left")
plt.show()
plt.plot(np.linspace(1, epochs, epochs), trainingerror, "r", label="Training Error")
plt.plot(np.linspace(1, epochs, epochs), validerror, label="Validation Error")
plt.xlabel("Epoch No.")
plt.ylabel("Error")
plt.title("Convergence History")
plt.legend(loc="upper left")
plt.show()




# testing
import numpy as np
import pandas as pd
#from main import *

df = pd.read_excel("Folds5x2_pp.xlsx", sheet_name="Sheet1")
df.dropna(axis=0, inplace=True)

Xtest = [[], [], [], []]
Ytest = [[]]
Xtest[0] = df.iloc[:957, 0]
Xtest[1] = df.iloc[:957, 1]
Xtest[2] = df.iloc[:957, 2]
Xtest[3] = df.iloc[:957, 3]
Ytest[0] = df.iloc[:957, 4]
Ytest = np.array(Ytest)
xtest = normalize(Xtest, 1)
xtest = np.array(xtest)
ytest = normalize(Ytest, 0.9)

ytestoutput = xtest
for layer in range(2):
    v = np.dot(w[layer], ytestoutput)
    ytestoutput = phi(v)
etest = ytest - ytestoutput
etest = np.divide(np.absolute(etest), np.add(np.absolute(ytest), 0.001))
error = np.sum(etest)*100 / len(ytestoutput[0])
print(f"MAPEtest = {error}")
