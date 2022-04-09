# Two-Layer-Neural-Network-for-MNIST
Use numpy to bulid a two-layer neural network for image classification based on MNIST.

### System Architecture 
Python 3.8

## Data Processing
* The MNIST data is seperated into train set and test set in `mnist_train.csv` and `mnist_test.csv`
* It is quite slow to read in the data from the csv files. We save the data in binary format with the dump function from the pickle module.
* Be sure that the dumped data `pickled_mnist.pkl` is at:
`./data/pickled_mnist.pkl`


## Training:
Training with the defalut hyper-parameters:

`python train_model.py`

For training with different hyper-parameters:

`python train_model.py -h 256 -lr 1e-2 -r 1e-4 -sa 0`

The args are as follows:
* `-h`:  `--hiddensize`, number of nodes in the hidden layer
* `-lr`:  `--learningrate`, learning rate 
* `-r`: `--regularization`, L2 regularization parameter 
* `-sa`: `--save`, 0: not save; 1: save the model. Note that if it is set to 1, this new model will be tested in testing.

For instance, the above command will train a network with size 784x256x10, learning rates for each layer 1e-2,  and regularization lambda 1e-4 witout saving the model.



## Testing:
Save the trained model as `mnist_model.pkl`and then

`python test.py `

* It will print the testing accuracy. 

* Download Trained Model at https://pan.baidu.com/s/1QLIp24xQJUrKIpwHFSaOAg pw:4qo3

## Results
The best accuracy was obtained using the following configuration:

* Input layer - 784 neurons
* Hidden layer - 512 neurons
* Output layer - 10 neurons
* Batch size – 64
* Epoch – 50
* Learning rate - 0.09
* Regularization parameter (L2):  
* Activation in intermediate Layers – ReLu 
* Activation in output Layer – Softmax
* Parameter initialization – He Normalization
* Training accuracy : 99.99%
* Test accuracy : 98.26%
