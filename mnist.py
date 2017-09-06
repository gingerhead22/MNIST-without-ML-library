from __future__ import print_function
import os
import numpy as np
import math

np.set_printoptions(precision = 5, suppress = True)

TRAINING_IMAGE='./data/train-images.idx3-ubyte'
TRAINING_LABEL='./data/train-labels.idx1-ubyte'
TEST_IMAGE='./data/t10k-images.idx3-ubyte'
TEST_LABEL='./data/t10k-labels.idx1-ubyte'
PIXELS=28*28
TRAINING_SIZE=60000
TEST_SIZE=10000

def load_data():
	#data loading and preprocessing
	trainImgFile=open(TRAINING_IMAGE)
	trainImg=trainImgFile.read()
	trainImg=trainImg[16:]
	trainX=[]
	for s in trainImg:
		trainX.append(ord(s[0]))
	trainX=np.array(trainX).reshape(TRAINING_SIZE,PIXELS)
	print("training images loaded, "+str(trainX.shape[0])+" datasets in total")

	trainLblFile=open(TRAINING_LABEL)
	trainLbl=trainLblFile.read()
	trainLbl=trainLbl[8:]
	trainY=[]
	for a in trainLbl:
		temp=[0]*10
		temp[ord(a[0])]=1
		trainY.append(temp)
	trainY=np.array(trainY).reshape(TRAINING_SIZE,10)
	print("training labels loaded, "+str(trainY.shape[0])+" datasets in total")

	testImgFile=open(TEST_IMAGE)
	testImg=testImgFile.read()
	testImg=testImg[16:]
	testX=[]
	for s in testImg:
		testX.append(ord(s[0]))
	testX=np.array(testX).reshape(TEST_SIZE,PIXELS)
	print("test images loaded, "+str(testX.shape[0])+" datasets in total")

	testLblFile=open(TEST_LABEL)
	testLbl=testLblFile.read()
	testLbl=testLbl[8:]
	testY=[]
	for a in testLbl:
		temp=[0]*10
		temp[ord(a[0])]=1
		testY.append(temp)
	testY=np.array(testY).reshape(TEST_SIZE,10)
	print("test labels loaded, "+str(testY.shape[0])+" datasets in total")

	valX=trainX[int(TRAINING_SIZE*0.8):]
	trainX=trainX[:int(TRAINING_SIZE*0.8)]
	valY=trainY[int(TRAINING_SIZE*0.8):]
	trainY=trainY[:int(TRAINING_SIZE*0.8)]

	return trainX, trainY, testX, testY, valX, valY

#model creating
class NN:
	ALPHA =  0.5
	LAMBDA = 1
	def __init__(self, num_input = PIXELS, num_output = 10, batch_size = 500):
		self.num_input = num_input
		self.num_hidden = 100
		self.num_output = num_output
		self.batch_size = batch_size
		self. W1 = np.random.randn(self.num_hidden*(self.num_input + 1)).reshape(self.num_input + 1, self.num_hidden)
		self.W2 = np.random.randn(self.num_output*(self.num_hidden + 1)).reshape(self.num_hidden + 1, self.num_output)
		self.valStack = []
	def sigmoid(self,z):
		m,n = z.shape
		ans = np.zeros(m*n).reshape(m,n)
		for i in range(m):
			for j in range(n):
				if z[i,j] < -100:
					z[i,j] = -100
				ans[i,j] = 1.0/(1+math.exp(-1*z[i,j]))
		return ans

	def normalize(self,z):
		m,n = z.shape
		b =np.random.randn(m*n).reshape(m,n)
		for i in range(m):
			b[i] = z[i]*1.0/np.sum(z[i])
		return b

	def add_bias(self,X):
		ONE = np.ones(X.shape[0]).reshape(X.shape[0],1)
		return np.hstack((ONE, X))

	def feed_forward(self,X):
		A1=self.add_bias(X)
		A2=self.sigmoid(np.dot(A1, self.W1))
		#add bias to hidden layer
		A2=self.add_bias(A2)
		Y=self.normalize(self.sigmoid(np.dot(A2, self.W2)))
		return Y, A2

	def cal_loss(self,Y, target):
		m,n = Y.shape
		sum = np.sum((Y-target)**2)
		return sum*1.0/m

	def cal_acc(self, Y, TARGET):
		base = Y.shape[0]
		y = np.argmax(Y,axis = 1)
		target = np.argmax(TARGET,axis = 1)
		result = np.count_nonzero((y == target))
		return result * 1.0/base

	def initialize(self,DELTA):
		m,n = DELTA.shape
		return np.zeros(m*n).reshape(m,n)

	def update_weights(self,DELTA1, DELTA2,SIZE):
		self.W1-=self.ALPHA/SIZE*(DELTA1)
		self.W2-=self.ALPHA/SIZE*(DELTA2)

	def validate(self,valX, valY):
		SIZE = valX.shape[0]
		CHOICE = SIZE/10
		order = np.random.choice(SIZE, CHOICE, replace = False)
		valx = valX[order]
		valy = valY[order]
		y, a2 = self.feed_forward(valx)
		loss = self.cal_loss(y, valy)
		acc = self.cal_acc(y, valy)
		if not self.valStack:
			self.valStack.append(loss)
		elif loss > self.valStack[-1]:
			self.valStack.append(loss)
		else:
			self.valStack = []
		print("val_acc: ",acc," val_loss: ",loss)

	def cease_training(self):
		if len(self.valStack) >=4:
			return True
		return False

	def train(self,trainX, trainY, valX, valY):
		SIZE = trainX.shape[0]
		iter = SIZE/self.batch_size
		epoch =0
		while(not self.cease_training()):
			loss = 0
			acc = 0
			epoch+=1
			DELTA1 = self.initialize(self.W1)
			DELTA2 = self.initialize(self.W2)
			for i in range(iter):
				X = trainX[i*self.batch_size:(i+1)*self.batch_size]
				target = trainY[i*self.batch_size:(i+1)*self.batch_size]
				Y, A2= self.feed_forward(X)
				loss+= self.cal_loss(Y, target)*1.0/iter
				acc+= self.cal_acc(Y, target)*1.0/iter
				theta3=Y-target
				ONE=np.ones(A2.shape[0]*A2.shape[1]).reshape(A2.shape[0],A2.shape[1])
				index=np.multiply(A2,ONE-A2)
				theta2=np.multiply(np.dot(theta3, self.W2.transpose()),index)
				DELTA2+=np.dot(A2.transpose(), theta3)
				theta2=theta2[:,1:]
				A1=self.add_bias(X)
				DELTA1+=np.dot(A1.transpose(), theta2)
				if i%10 ==0:
					self.update_weights(DELTA1, DELTA2,10*self.batch_size)
					DELTA2 = self.initialize(DELTA2)
					DELTA1 = self.initialize(DELTA1)
			if epoch%2 == 0:
				print("############# epoch ",epoch," #############")
				print("acc: ",acc," loss: ",loss)
				self.validate(valX, valY)
				print()

		print("######### training finished #############")
		return True

	def evaluate(self,testX, testY):
		SIZE = testX.shape[0]
		iter = SIZE/self.batch_size
		LOSS = 0
		ACC = 0
		for i in range(iter):
			X = testX[i*self.batch_size:(i+1)*self.batch_size]
			target = testY[i*self.batch_size:(i+1)*self.batch_size]
			Y, A2 = self.feed_forward(X)
			LOSS +=self.cal_loss(Y, target)*1.0/iter
			ACC +=self.cal_acc(Y, acc)*1.0/iter
		print("########### evaluation ############")
		print("test_acc: ", ACC, "test_loss: ", LOSS)


def main():
	trainX, trainY, testX, testY, valX, valY = load_data()
	print(trainX.shape, trainY.shape, testX.shape, testY.shape, valX.shape, valY.shape)
	nn = NN(PIXELS, 10, 500)
	nn.train(trainX, trainY, valX, valY)
	nn.evaluate(testX, testY)


if __name__ == "__main__":
	main()
