from __future__ import print_function
import os
import numpy as np 
import math

#data loading and preprocessing
TRAINING_IMAGE='./data/train-images.idx3-ubyte'
TRAINING_LABEL='./data/train-labels.idx1-ubyte'
TEST_IMAGE='./data/t10k-images.idx3-ubyte'
TEST_LABEL='./data/t10k-labels.idx1-ubyte'
PIXELS=28*28
TRAINING_SIZE=60000
TEST_SIZE=10000

trainImgFile=open(TRAINING_IMAGE)
trainImg=trainImgFile.read()
trainImg=trainImg[16:]
trainX=[]
for s in trainImg:
	trainX.append(ord(s[0]))
trainX=np.array(trainX).reshape(TRAINING_SIZE,PIXELS)
print(trainX.shape)
print("training images loaded, "+str(trainX.shape[1])+" datasets in total")

trainLblFile=open(TRAINING_LABEL)
trainLbl=trainLblFile.read()
trainLbl=trainLbl[8:]
trainY=[]
for a in trainLbl:
	temp=[0]*10
	temp[ord(a[0])]=1
	trainY.append(temp)
trainY=np.array(trainY).reshape(TRAINING_SIZE,10)
print(trainY.shape)
print("training labels loaded, "+str(trainY.shape[1])+" datasets in total")

testImgFile=open(TEST_IMAGE)
testImg=testImgFile.read()
testImg=testImg[16:]
testX=[]
for s in testImg:
	testX.append(ord(s[0]))
testX=np.array(testX).reshape(TEST_SIZE,PIXELS)
print(testX.shape)
print("test images loaded, "+str(testX.shape[1])+" datasets in total")

testLblFile=open(TEST_LABEL)
testLbl=testLblFile.read()
testLbl=testLbl[8:]
testY=[]
for a in testLbl:
	temp=[0]*10
	temp[ord(a[0])]=1
	testY.append(temp)
testY=np.array(testY).reshape(TEST_SIZE,10)
print(testY.shape)
print("test labels loaded, "+str(testY.shape[1])+" datasets in total")

valX=trainX[int(TRAINING_SIZE*0.8):]
trainX=trainX[:int(TRAINING_SIZE*0.8)]
valY=trainY[int(TRAINING_SIZE*0.8):]
trainY=trainY[:int(TRAINING_SIZE*0.8)]


#model creating
def sigmoid(z):
	h=[]
	for i in range(z.shape[0]):
		h.append(1.0/(1+math.exp(-1*z[i])))
	return np.array(h)

R2=100
W1=np.random.rand(R2,PIXELS+1)
#B1=np.random.rand(R2,1)
W2=np.random.rand(10,R2+1)
#B2=np.random.rand(10,1)
ALPHA=0.5
LAMBDA=1.0
M=trainX.shape[0]

def loss_func(Y, target):
	K=len(Y)
	sum=0
	for i in range(K):
		sum+=(target[i]-Y[i])**2
		#sum+=target[i]*math.log(Y[i])+(1-target[i])*math.log(1-Y[i])
	return sum

def item_square(W):
	sum = 0
	for vec in W:
		for item in vec:
			sum+=item**2
	return sum

def process(W1,W2,X,Y):
	m=X.shape[0]
	loss=0
	for i in range(m):
		target=Y[i]
		x=np.append(np.ones(1),X[i])
		A2=sigmoid(np.dot(W1,x))
		#add bias to hidden layer
		A2=np.append(np.ones(1),A2)
		y=sigmoid(np.dot(W2,A2))
		loss+=loss_func(y,target)
	loss=loss*(-1.0)/m+LAMBDA/(2*m)*(item_square(W1)+item_square(W2))
	return loss



iteration=0
MAX=1000

def stop_training(stack):
	if len(stack)>2:
		return True
	else:
		return False

val_stack=[]
while(1):
	iteration+=1
	DELTA1=np.zeros(R2*(PIXELS+1)).reshape(R2,PIXELS+1)
	DELTA2=np.zeros(10*(R2+1)).reshape(10,R2+1)
	loss=0
	for i in range(M):
		#pick ith training data and add bias
		X=np.append(np.ones(1),trainX[i])
		target=trainY[i]

		A2=sigmoid(np.dot(W1,X))
		#add bias to hidden layer
		A2=np.append(np.ones(1),A2)
		Y=sigmoid(np.dot(W2,A2))
		
		theta3=Y-target
		ONE=np.ones(A2.shape[0])
		index=np.multiply(A2,ONE-A2)
		theta2=np.multiply(np.dot(W2.transpose(),theta3),index)
		theta3=theta3.reshape(theta3.shape[0],1)
		A2=A2.reshape(A2.shape[0],1)
		DELTA2+=np.dot(theta3,A2.transpose())
		theta2=theta2.reshape(theta2.shape[0],1)
		theta2=theta2[1:]
		X=X.reshape(X.shape[0],1)
		DELTA1+=np.dot(theta2,X.transpose())
		loss+=loss_func(Y,target)
	loss=loss*(-1.0)/M
	loss+=LAMBDA/(2*M)*(item_square(W1)+item_square(W2))
	W1-=1.0/M*(DELTA1+LAMBDA*W1)
	W2-=1.0/M*(DELTA2+LAMBDA*W2)
	W1[:,0]+=LAMBDA/M*W1[:,0]
	W2[:,0]+=LAMBDA/M*W2[:,0]
	if iteration%30==0:
		print("iteration: ", iteration)
		val_loss=process(W1,W2,valX,valY)
		print("loss: ", loss,"val_loss: ", val_loss)
		print("#############################################")
		if len(val_stack) == 0 or val_loss>val_stack[-1]:
			val_stack.append(val_loss)
		else:
			val_stack=[val_loss]
	if stop_training(val_stack) or iteration>MAX:
		break


