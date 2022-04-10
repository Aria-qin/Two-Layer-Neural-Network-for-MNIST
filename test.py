import pickle
from train_model import *
def main():
    print("导入数据完毕")
    # 导入储存的模型
    with open("./mnist_model.pkl", "br") as fh:
        model = pickle.load(fh)
    # forward pass
    model.forward(X_test, y_test)
    # test accuracy
    acc = np.count_nonzero(np.argmax(model.a2,axis=1) == np.argmax(y_test,axis=1)) / X_test.shape[0]
    print("Test Accuracy:", 100 * acc, "%")

if __name__ == '__main__':
    main()


