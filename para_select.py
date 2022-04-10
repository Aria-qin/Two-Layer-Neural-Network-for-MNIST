'''
Author: your name
Date: 2022-04-09 13:06:32
LastEditTime: 2022-04-10 20:15:22
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /hw1/para_select copy.py
'''
from train_model import *

# 初始设定中，层数为784*512*10
# 尝试不同的层数设定：
# 首先尝试 128, 256, 512三个值，找到一个较优的大致范围
hidden_size_first_try = [128, 256, 512]
best_hidden = 0
for i in range(len(hidden_size_first_try)):
    NN = Nerual_Network(784,hidden_size_first_try[i], 10, 1e-2, 1e-4, epoch = 30) 
    best = NN.train(X_train, y_train)
    if best>best_hidden:
        best_hidden = best
        best_ind = i
    print('hiddenlayer:', hidden_size_first_try[i],  'best accuracy: ', best)

print(hidden_size_first_try[best_ind])  

# best_hidden = 96.91168623265742
# 根据上面的探索，隐藏层大小 512 时效果最好。在512附近随机取6个点
hidden_size_list = np.random.randint(low=480, high=540, size=6)
for i in range(len(hidden_size_list)):
    NN = Nerual_Network(784,hidden_size_list[i], 10, 1e-2, 1e-4, epoch = 30) 
    best = NN.train(X_train, y_train)
    if best>best_hidden:
        best_hidden = best
        best_ind = i
    print('hiddenlayer:', hidden_size_list[i],  'best accuracy: ', best)
 
# 根据上面的探索，层数 784*512*10效果最好（与515效果非常接近）
# 然后从对数尺度上对学习率和正则化参数做一个初步的探索
learning_rates = np.array([1, 10, 100, 1000])*1e-4
for lr in learning_rates:
    NN = Nerual_Network(784,512,10,lr, 1e-4, epoch = 30) 
    best = NN.train(X_train, y_train)
    print('lr:', lr,  'best accuracy: ', best)

# 根据上面的搜索，学习率在0.1左右表现最好，并且训练准确率已经达到 99.9%，
# 可能出现过拟合，对正则化参数进行搜索，并且选取test set 前1000个样本作为validation set
# learning_rates1 = np.random.uniform(0.07, 0.11, 4)
reg_strengths = np.array([1, 10, 100, 1000])*1e-6
for reg in reg_strengths:
    NN = Nerual_Network(784,512,10,0.1, reg, epoch = 30) 
    best = NN.train(X_train[0:1000], y_train[0:1000])
    print('reg:', reg,  'best accuracy: ', best)
    NN.test(X_test, y_test)

# 当 reg = 1e-4 时表现最好（并且与其他的正则化参数相比差别不大）

NN = Nerual_Network(784,512,10,0.1, 1e-4, test_data=(X_test,y_test), epoch = 30) 