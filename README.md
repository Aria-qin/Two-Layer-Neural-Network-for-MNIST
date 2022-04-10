# Two-Layer-Neural-Network-for-MNIST
Use numpy to bulid a two-layer neural network for image classification based on MNIST.

Download Trained Model at https://pan.baidu.com/s/1QLIp24xQJUrKIpwHFSaOAg pw:4qo3
### System Architecture 
Python 3.8

Main files:  `train_model.py`,  `para_tuning.py`,  `test.py`,  `mnist_model.pkl`

Supporting files:  `data_processing.py`,  `visualization.py`,  `parse.py`
## Data Processing
* The MNIST data is seperated into train set and test set in `mnist_train.csv` and `mnist_test.csv`
* It is quite slow to read in the data from the csv files. We save the data in binary format with the dump function from the pickle module.
* Be sure that the dumped data `pickled_mnist.pkl` is at:
`./data/pickled_mnist.pkl`


## Training:
Training with the defalut hyper-parameters:

`python train_model.py`

For training with different hyper-parameters:

`python train_model.py -h 256 -l 1e-2 -r 1e-4`

The args are as follows:
* `-h`:  `--hiddensize`, number of nodes in the hidden layer
* `-l`:  `--learningrate`, learning rate 
* `-r`: `--regularization`, L2 regularization parameter 


For instance, the above command will train a network with size 784x256x10, learning rates for each layer 1e-2,  and regularization lambda 1e-4.

## Parameter Tuning
The details of parameter tuning can ba found in `para_tuning.py`

## Testing:
* The training process will save the trained model as `mnist_model.pkl`and then 

 `python test.py `

* It will print the testing accuracy. 


## Results
The best accuracy was obtained using the following configuration:

* Input layer - 784 neurons
* Hidden layer - 512 neurons
* Output layer - 10 neurons
* Epoch – 50
* Learning rate - 0.1
* Regularization parameter (L2):  
* Activation in intermediate Layers – ReLu 
* Activation in output Layer – Softmax
* Parameter initialization – He Normalization
* Training accuracy : 99.99%
* Test accuracy : 98.26%
