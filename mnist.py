from __future__ import print_function
import os
import numpy as np 
import math


def load_data():
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
	print("training labels loaded, "+str(trainY.shape[1])+" datasets in total")

	testImgFile=open(TEST_IMAGE)
	testImg=testImgFile.read()
	testImg=testImg[16:]
	testX=[]
	for s in testImg:
		testX.append(ord(s[0]))
	testX=np.array(testX).reshape(TEST_SIZE,PIXELS)
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
	print("test labels loaded, "+str(testY.shape[1])+" datasets in total")

	valX=trainX[int(TRAINING_SIZE*0.8):]
	trainX=trainX[:int(TRAINING_SIZE*0.8)]
	valY=trainY[int(TRAINING_SIZE*0.8):]
	trainY=trainY[:int(TRAINING_SIZE*0.8)]

	return trainX, trainY, testX, testY, valX, valY

#model creating

class NN:
	ALPHA =  0.5
	LAMBDA = 1
	def __init__(self, num_input, num_output, batch_size):
		self.num_input = num_input
		self.num_hidden = 100
		self.num_output = num_output
		self.batch_size = batch_size
		self. W1 = np.random.randn(num_hidden*(num_input + 1)).reshape(num_hidden, num_input + 1)
		self.W2 = np.random.randn(num_output*(num_hidden + 1)).reshape(num_output, num_hidden + 1)

	def sigmoid(z):
		m,n = z.shape
		ans = []
		for j in range(m):
			h=[]
			for i in range(n):
				h.append(1.0/(1+math.exp(-1*z[i])))
			ans.append(h)
		return np.array(ans).reshape(m,n)

	def normalize(z):
		m,n = z.shape
		b =np.random.randn(m*n).reshape(m,n)
		for i in range(m):
			b[i] = z[i]*1.0/np.sum(z[i])
		return b

	def add_bias(X):
		ONE = np.ones(X.shape[0])
		temp = X.transpose
		Y=np.vstack((ONE, temp))
		return Y.transpose()

	def feed_forward(X):
		A1=self.add_bias(X)
		A2=self.sigmoid(np.dot(self.W1,A1))
		#add bias to hidden layer
		A2=self.add_bias(A2)
		Y=self.normalize(self.sigmoid(np.dot(self.W2,A2)))
		return Y, A2

	def loss_calculate(Y, target):
		m,n = Y.shape
		sum = np.sum((Y-target)**2)
		return sum

	def initialize(DELTA):
		m,n =DELTA.shape
		return np.zeros(m,n)

	def update_weights(DELTA1, DELTA2,SIZE):
		self.W1-=self.ALPHA/SIZE*(DELTA1)
		self.W2-=self.ALPHA/SIZE*(DELTA2)
	def validate():

	def print_loss():

	def cease_training():

	def train(trainX, trainY):
		SIZE = trainX.shape[0]
		iter = SIZE/self.batch_size
		loss = 0
		count =0
		While not self.cease_training():
			count+=1
			DELTA1 =
			DELTA2 =
			for i in range(iter):
				X = trainX[i*self.batch_size:(i+1)*self.batch_size]
				target = trainY[i*self.batch_size:(i+1)*self.batch_size]
				Y, A2= self.feed_forward(X)
				loss+= self.loss_calculate(Y, target)
				theta3=Y-target
				ONE=np.ones(A2.shape[0]*A2.shape[1]).reshape(A2.shape[0],A2.shape[1])
				index=np.multiply(A2,ONE-A2)
				theta2=np.multiply(np.dot(self.W2.transpose(),theta3),index)
				DELTA2+=np.dot(theta3,A2.transpose())
				theta2=theta2[1:]
				DELTA1+=np.dot(theta2,X.transpose())
					if i%10 ==0:
						self.update_weights(DELTA1, DELTA2,10*self.batch_size)
						DELTA2 = self.initialize(DELTA2)
						DELTA1 = self.initialize(DELTA1)
			loss = loss*1.0/SIZE
			if count%10 == 0:
				self.validate()
				self.print_loss()
		print("training finished")
		return True

	def evaluate(testX, testY):
		SIZE = testX.shape[0]

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
	sum = np.sum((Y-target)**2)
	return sum*1.0/

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
	loss = loss
	return loss



iteration=0
MAX=1000

def stop_training(stack):
	if len(stack)>2:
		return True
	else:
		return False

val_stack=[]
batchSize=100
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
	loss=loss/M
	W1-=1.0/M*(DELTA1)
	W2-=1.0/M*(DELTA2)
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


