# Two-Layer-Neural-Network-for-MNIST
Use numpy to bulid a two-layer neural network for image classification based on MNIST.

Download Trained Model at: https://pan.baidu.com/s/1cA3tq2qJR6_0FFPEIld6rA pwd: woen
### System Architecture 
Python 3.8

Main files:  `train_model.py`,  `para_select.py`,  `test.py`,  `mnist_train.csv`, `mnist_test.csv`

Supporting files:  `visualization.py`,  `parse.py`
## Data Processing
* The MNIST data is seperated into train set and test set in `mnist_train.csv` and `mnist_test.csv`
* It is quite slow to read in the data from the csv files. In practice, I save the data in binary format with the dump function from the pickle module, but it is too big to upload, so I just upload the csv files at   as well as the whole model.
* To run the code, make sure that the csv data is at `./data/`


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

This might be time-consuming, please wait several minutes for reading the data and training the first epoch.

## Parameter Tuning
The details of parameter tuning can ba found in `para_select.py`

## Testing:
* The training process will save the trained model as `mnist_model.pkl`and then 

 `python test.py `

* It will print the testing accuracy. Please wait about 1 minute for reading the data from csv file.


## Results
The best accuracy was obtained using the following configuration:

* Input layer: 784 neurons
* Hidden layer: 512 neurons
* Output layer: 10 neurons
* Epoch: 50
* Learning rate: 0.1
* Regularization parameter (L2):  1e-4
* Activation in intermediate Layers: ReLu 
* Activation in output Layer: Softmax
* Parameter initialization: He Normalization
* Training accuracy : 99.87%
* Test accuracy : 98.17%
