from train_model import Nerual_Network
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import pickle
style.use('ggplot')

def img_show(W, height,width, plot_size, save=False, filename = ''):
    m, n = plot_size
    fig, axes = plt.subplots(m, n)
    plt.subplots_adjust(wspace=0.05,hspace=0.05)
    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = W.min(), W.max()
    for col, ax in zip(W.T, axes.ravel()):
        ax.matshow(col.reshape(height,width), cmap=plt.cm.gray, vmin=0.5*vmin, vmax=0.5*vmax)
        ax.set_xticks(())
        ax.set_yticks(())
    # plt.show()

with open("./mnist_model.pkl", "br") as fh:
    model = pickle.load(fh)


train_loss = model.train_loss
train_accuracy = model.train_accuracy
test_accuracy = model.test_accuracy
train_loss = model.train_loss
test_loss = model.test_loss

def plot_acc(train_accuracy, test_accuracy):
    plt.plot(train_accuracy,'o-', markersize = 3)
    plt.plot(test_accuracy,'^-', markersize = 3)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(89,101)
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy.png')

def plot_loss(train_loss, test_loss):
    plt.plot(train_loss,'o-', markersize = 3)
    plt.plot(test_loss,'^-', markersize = 3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.ylim(89,101)
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss.png')


img_show(model.W1, 28, 28, (10, 10))
img_show(model.W2, 32,16, (2,5))
img_show(np.dot(model.W1, model.W2), 28, 28, (2, 5))
plot_loss(train_loss, test_loss)






