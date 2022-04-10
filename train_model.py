import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import sys

# activation_function:  ReLU
def ReLU(x):
        return np.maximum(0,x)
# derivative of ReLU
def der_ReLU(x):
        return 1 * (x > 0) 

#Softmax
def softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(z)/np.sum(np.exp(z), axis=-1, keepdims=True)
    
class Nerual_Network(object):
    def __init__(self, input_layer, hidden_layer,  output_layer, learningrate, reg,  test_data = False, lr_decay = 0.95, epoch = 50):
        """
        :param input_layer: 输入层结点数
        :param hidden_layer: 隐藏层结点数
        :param output_layer: 输出层结点数
        :param learningrate: 学习率
        :param reg: 正则化参数
        :param lr_decay: 学习率衰减因子
        :param epoch: epoch数量
        """
        self.inputnodes = input_layer
        self.hiddennodes = hidden_layer
        self.outputnodes = output_layer
        self.learningrate = learningrate
        self.lr = learningrate  # store the initial learning rate
        self.reg = reg
        self.lr_decay = lr_decay # decay rate per epoch
        self.epoch = epoch
        self.train_loss = []
        self.test_loss = []
        self.train_accuracy = []
        self.test_data = test_data
        self.test_accuracy =[]
        self.initialize_matrix()
        
    
    def initialize_matrix(self):
        self.W1 = np.random.normal(0, np.sqrt(2/self.inputnodes), (self.inputnodes, self.hiddennodes))
        self.W2 = np.random.normal(0, np.sqrt(2/self.hiddennodes), (self.hiddennodes, self.outputnodes))
        
        self.b1 = np.random.normal(0, np.sqrt(2/self.inputnodes), (self.hiddennodes,))
        self.b2 = np.random.normal(0, np.sqrt(2/self.hiddennodes), (self.outputnodes,))


    def forward(self, input_vector, output_vector):
        self.z1 = np.dot(input_vector, self.W1) + self.b1
        self.a1 = ReLU(self.z1)   #sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)   #sigmoid(self.z2)
        
        self.error = self.a2 - output_vector
        
    def backpropagation(self, input_vector):
        #  derivative of data loss
        d_data_loss = self.error

        dW2 = np.dot(d_data_loss.T,self.a1).T+ self.reg * self.W2
        db2 = np.sum(d_data_loss,axis = 0,keepdims=True)

        dh = np.dot(d_data_loss, self.W2.T)*der_ReLU(self.z1)
        dW1 = np.dot(dh.T,input_vector).T+ self.reg * self.W1
        db1 = np.sum(dh,axis = 0,keepdims=True)

    
        self.W2 = self.W2 - self.learningrate * dW2
        self.W1 = self.W1 - self.learningrate * dW1
        
        self.b2 = self.b2 - self.learningrate * db2
        self.b1 = self.b1 - self.learningrate * db1


    def train(self, X, Y):
        num_train = X.shape[0]
        best_acc = 0
        for epoch in range(self.epoch):
            idx = [i for i in range(num_train)]
            np.random.shuffle(idx)
            X = X[idx]
            Y = Y[idx]
            for iter in range(num_train):
                X_train = X[iter].reshape(1, 784)
                Y_train = Y[iter].reshape(1,10)
                self.forward(X_train, Y_train)
                self.backpropagation(X_train)

            if self.test_data:
                x_test, y_test = self.test_data
                self.forward(x_test, y_test)
                test_acc = np.count_nonzero(np.argmax(self.a2,axis=1) == np.argmax(y_test,axis=1)) / x_test.shape[0]
                self.test_loss.append(np.mean(self.error**2)+1/2*self.reg*(np.sum(self.W1*self.W1)+np.sum(self.W2*self.W2)))
                self.test_accuracy.append(test_acc*100)

            self.forward(X, Y)
            train_acc = np.count_nonzero(np.argmax(self.a2,axis=1) == np.argmax(Y,axis=1)) / num_train
            self.train_loss.append(np.mean(self.error**2)+1/2*self.reg*(np.sum(self.W1*self.W1)+np.sum(self.W2*self.W2)))
            self.train_accuracy.append(train_acc*100)

            self.learningrate *= self.lr_decay
    
            print('epoch: ', epoch, "train loss:", self.train_loss[epoch], "train acc: ", self.train_accuracy[epoch])
            if best_acc < self.train_loss[epoch]:
                best_acc = self.train_loss[epoch]
        return best_acc
    
    def test(self, X_test, y_test):
        self.forward(X_test, y_test)
        acc = np.count_nonzero(np.argmax(self.a2,axis=1) == np.argmax(y_test,axis=1)) / X_test.shape[0]
        print("Test Accuracy:", 100 * acc, "%")


def load_data(path):
    def one_hot(y):
        table = np.zeros((y.shape[0], 10))
        for i in range(y.shape[0]):
            table[i][int(y[i][0])] = 1 
        return table

    def normalize(x): 
        x = x / 255*0.99+0.01
        return x 

    data = np.loadtxt('{}'.format(path), delimiter = ',')
    return normalize(data[:,1:]),one_hot(data[:,:1])

X_train, y_train = load_data('./data/mnist_train.csv')
X_test, y_test = load_data('./data/mnist_test.csv')

def save_model(nn):
    pickle.dump(nn, open('mnist_model_new.pkl', 'wb'))

if __name__ == '__main__':
    from parse import readCommand
    args = readCommand(sys.argv[1:]) #Read Arguments
    NN = Nerual_Network(**args) 
    NN.train(X_train, y_train)
    # save model
    # save_model(NN)