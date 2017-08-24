from __future__ import print_function
import os
import numpy as np 

#data loading and preprocessing
TRAINING_IMAGE='./data/train-images.idx3-ubyte'
TRAINING_LABEL='./data/train-labels.idx1-ubyte'
TEST_IMAGE='./data/t10k-images.idx3-ubyte'
TEST_LABEL='./data/t10k-labels.idx1-ubyte'
PIXELS=28*28
TRAINING_DATASETS=60000
TEST_DATASETS=10000

trainImgFile=open(TRAINING_IMAGE)
trainImg=trainImgFile.read()
trainImg=trainImg[16:]
trainX=[]
for s in trainImg:
	trainX.append(ord(s[0]))
trainX=np.array(trainX).reshape(TRAINING_DATASETS,PIXELS)
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
trainY=np.array(trainY).reshape(TRAINING_DATASETS,10)
print(trainY.shape)
print ("training labels loaded, "+str(trainY.shape[1])+" datasets in total")

testImgFile=open(TEST_IMAGE)
testImg=testImgFile.read()
testImg=testImg[16:]
testX=[]
for s in testImg:
	testX.append(ord(s[0]))
testX=np.array(testX).reshape(TEST_DATASETS,PIXELS)
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
testY=np.array(testY).reshape(TEST_DATASETS,10)
print(testY.shape)
print ("test labels loaded, "+str(testY.shape[1])+" datasets in total")

#model creating


def sigmoid(z):
	h=[]
	for i in range(z.shape[0]):
		h.append(1.0/(1+math.exp(-1*z[i])))
	return np.array(h)

R2=PIXELS
W1=np.random.rand(R2,PIXELS+1)
#B1=np.random.rand(R2,1)
W2=np.random.rand(10,R2+1)
#B2=np.random.rand(10,1)
Y=np.zeros(10,1)
A2=np.zeros(R2,1)

while(1):
	DELTA2=np.zeros(R2, PIXELS+1)
	DELTA3=np.zeros(10,R2+1)
	#BDELTA2=np.zeros(R2,1)
	#BDELTA1=np.zeros(10,1)
	for i in range(trainX.shape[0]):
		#pick ith training data and add bias
		X=np.concatenate(np.ones(1),trainX[i])
		target=trainY[i]

		A2=sigmoid(np.dot(W1,X))
		#add bias to hidden layer
		A2=np.concatenate(np.ones(1),A2)
		Y=sigmoid(np.dot(W2,A2))
		
		#learning rate
		ALPHA=0.3
		LAMBDA=1

		delta3=Y-target

		index=np.multiply(A2,(np.ones(A2.shape[0],A2.shape[1])-A2))
		delta2=np.multiply(np.dot(W2.transpose(),delta3),index)

		DELTA3+=

