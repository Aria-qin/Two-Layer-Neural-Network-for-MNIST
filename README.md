# Two-Layer-Neural-Network-for-MNIST
Use numpy to bulid a two-layer neural network for image classification based on MNIST.
## Data Processing
* The MNIST data is seperated into train set and test set in `mnist_train.csv` and `mnist_test.csv`
* It is quite slow to read in the data from the csv files. We save the data in binary format with the dump function from the pickle module.
* Be sure that the dumped data `pickled_mnist.pkl` is at:
`./data/pickled_mnist.pkl`

## Training:
Training with the defalut hyper-parameters:

`python train_model.py`

For training with different hyper-parameters:

`python train_model.py -s1 256 -s2 128 -lr1 1e-3 -lr2 1e-3 -lr3 1e-3 -r 1e-4`

Note that our model has two hidden layers, and there are in total 6 hyper-parameters:
* `-s1`:  `--hiddensize1`, number of nodes in the first hidden layer
* `-s2`:  `--hiddensize2`, number of nodes in the second hidden layer
* `-lr1`:  `--learningrate1`, learning rate between the input layer and first hidden layer
* `-lr2`:  `--learningrate2`, learning rate between the first hidden layer and the second hidden layer
* `-lr3`:  `--learningrate3`, learning rate between the second hidden layer and the output layer
* `-r`: `--regularization`, L2 regularization parameter 

For instance, the above command will train a network with size 784x256x128x10, learning rates for each layer 1e-3,  and regularization lambda 1e-4.



## Testing:
Save the trained model as `mnist_model.pkl`and then

`python test.py `

* It will print the testing accuracy. 

* Download Trained Model at https://pan.baidu.com/s/1QLIp24xQJUrKIpwHFSaOAg pw:4qo3
