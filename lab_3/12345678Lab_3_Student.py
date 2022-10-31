#!/usr/bin/env python
# coding: utf-8

# # Lab 3
# 
# Welcome to your third lab! This notebook contains all the code and comments that you need to submit. Labs are two weeks long and the places where you need to edit are highlighted in red. Feel free to add in your own markdown for additional comments.
# 
# __Submission details: make sure you have run all your cells from top to bottom (you can click _Kernel_ and _Restart Kernel and Run All Cells_). Submit this Jupyter Notebook and also submit the .py file that is generated.__

# In[3]:


## This code snippet does not need to be edited
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
from python_environment_check import check_packages
from python_environment_check import set_background

## Colour schemes for setting background colour
white_bgd = 'rgba(0,0,0,0)'
red_bgd = 'rgba(255,0,0,0.2)'


# In[4]:


## Code snippets in red (similar to this) is where you need to edit your answer)
# Set your student ID and name here:

student_number = 12345678  # 12345678
student_name = "John Doe" # "John Doe"

set_background(red_bgd)


# ## What you are going to do in this lab exercise!
# 
# In this laboratory exercise, you will create algorithms to train non-linear models using MLPs. 
# 
# In all of the tasks below, you should use PyTorch Lightning to design, train and test MLP models. You can use any pre-written libraries like matplotlib, pandas and numpy to visualize your results. 
# 
# <div class="alert alert-block alert-warning">
#     
# - **Section 1** :  Approximate the sine function.
# - **Section 2** : Design a shallow MLP for the fashion MNIST
# - **Section 3** : Comparing the performance of Shallow MLP vs Deep MLP.
#     
# </div>

# ## Section 1 : Approximate the sine function

# <div class="alert alert-block alert-info">
# 
# ## What you should do in this task!
# 
# In this section, we will approximate the sine function with a neural network to get a sense of how architecture and hyperparameters affect neural network performance. 
#     
# #### In this task, you will work on the following points:
#  1. As the first step you should have PyTorch and PyTorch-Lightning installed in your local machine. We have given you the set of necessary libraries and details on PyTorch functionalities that you need to use when implementing algorithm for Task 1. The first sub task is to create your custom dataset using PyTorch datasets and dataloaders.
#     
#  2. Define the custom dataset class (i.e., Train and Test datasets) and visualize the train dataset you've created.
#     
#  3. Design the Shallow Linear MLP model using PyTorch Lightning Module.
#     
#  4. Train and evaluate the MLP model on defined train dataset and test dataset.
#     
#  5. Visualize experimental results using Matplotlib.
# 
# 
# Before approaching, we will be introducing Pytorch Lightning Neural Network structure, Pytorch datasets, dataloaders and optimizers.
# 
# <img src="./figures/sine_wave.gif" width="1200" align="center">

# ## PyTorch & PyTorch-Lightning Installation
# 
# What are the differences between PyTorch and PyTorch-Lightning?
# 
# PyTorch is based on the Torch library, adapted for Python. PyTorch is a deep learning library that allows you to have greater control of your neural network architecture, and have a lot more customisable hyper parameters / functions (such as the loss function etc.) compared to other deep learning frameworks such as TensorFlow / Keras.
# 
# PyTorch-Lightning is a higher level of PyTorch, meaning that it is easier to use and run models on compared to the original PyTorch. There are methods that are already built into the classes for PyTorch-Lightning which you don't have to worry about as much, and takes away a lot of the additional considerations such as passing your dataset via CUDA, hence PyTorch-Lightning code will look simpler than PyTorch!
# 
# As the first step we import all the necessary libraries needed to implement these lab tasks. 
# 
# Note: Do not modify seed values or use any other libraries.

# In[5]:


# If you run on Jupyter Lab uncomment bellow comment
#! pip install --quiet "matplotlib" "pytorch-lightning" "pandas" "torchmetrics" 
#! pip install --ignore-installed "PyYAML"

# If you run on google colab uncomment bellow comment
#! pip install --quiet "matplotlib" "pytorch-lightning" "pandas" "torchmetrics"
import numpy as np
import random
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch 
import torch.nn as nn 
import torchmetrics
from torchmetrics import Accuracy

# For reproducability
torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

# Define GPU number
GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')


# <h2> Pytorch Datasets and Dataloaders </h2>
# 
# Pytorch has a huge number of functionalities that make training our neural networks very easy! One of those functionalities is the Pytorch dataset and dataloader (they are real life-savers!). In depth review on PyTorch Datasets and Dataloaders are covered in Lectures!

# In order to load datasets, we will use torchvision module and to create a dataloader, we will use sub module of torch as follows.
# 
# **from torch.utils.data import DataLoader**
# 
# Under torchvision datasets you can find popular datasets that are frequently used for machine learning/deep learning tasks (eg., MNIST, SVHN, CIFAR10, CIFAR100 etc). You can create your own custom dataset using torchvision dataset in built functionalities and in this task we will show how to load custom datasets.

# In[6]:


from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


# ### 1.1 Creating a Pytorch dataset
# 
# The dataset you are going to be creating will be points from a "noisy" sine wave. To create the custom dataset, you can use PyTorch dataset inbuilt functionalities. <br>
# 
# 
# The Pytorch dataset class has three essential parts:<br>
# 1. The \__init__ function (as most Python classes do)<br>
# 2. The \__getitem__ function (this is called during every iteration)<br>
# 3. The \__len__ function (this must return the length of the dataset)
# 
# **Remember! The "self" within classes will become attributes of that class that you can use within other methods that are defined for that class. If you defined an attribute without self.<\name>, then that attribute cannot be used for other methods.**
# 
# Make sure to follow all inline comments specified for each code fragment that you need to work on.

# In[7]:


set_background(red_bgd)

# Create a "SineDataset" class by importing the Pytorch Dataset class

class SineDataset(Dataset):
    """ Data noisy sinewave dataset
        num_datapoints - the number of datapoints you want
    """
    def __init__(self, num_datapoints):
        # Lets generate the noisy sinewave points
        
        # Create "num_datapoints" worth of random x points using a uniform distribution (0-1) using torch.rand
        # Then scale and shift the points to be between -9 and 9
        self.x_data = (torch.rand(num_datapoints,1) - 0.5) * 18
        
        # Calculate the sin of all data points in the x vector and the scale amplitude
        # Scale the amplitude by diving by 2.5
        self.y_data = torch.sin(self.x_data) / 2.5
        
        # Add some gaussein noise to each datapoint using torch.randn_like
        # Note:torch.randn_like will generate a tensor of gaussein noise the same size 
        # and type as the provided tensor
        # Divide the randomly generated values by 20 (so it is less noisy)
        self.y_data += torch.randn_like(self.y_data) / 20

    def __getitem__(self, index):
        # This function is called by the dataLOADER class whenever it wants a new mini-batch
        # The dataLOADER class will pass the dataSET and number of datapoint indexes (mini-batch of indexes)
        # It is up to the dataSET's __getitem__ function to output the corresponding input datapoints 
        # AND the corresponding labels
        return self.x_data[index], self.y_data[index]
        # Note:Pytorch will actually pass the __getitem__ function one index at a time
        # If you use multiple dataLOADER "workers" multiple __getitem__ calls will be made in parallel
        # (Pytorch will spawn multiple threads)

    def __len__(self):
        # We also need to specify a "length" function, Python will use this fuction whenever
        # You use the Python len(function)
        # We need to define it so the dataLOADER knows how big the dataSET is!
        return self.x_data.shape[0]


# ### 1.2 Create instances from defined custom dataset and visualize the training dataset
# 
# Now that you've defined your dataset, lets create an instance of it for training and testing and then create dataloaders to make it easy to iterate.

# In[8]:


set_background(red_bgd)

n_x_train = 30000   # the number of training datapoints
n_x_test = 8000     # the number of testing datapoints
BATCH_SIZE = 16     # the batch size for task 1

# Create an instance of the SineDataset for both the training and test set 
# (Here we have only training and test set. Therefore consider validation set also equals to test set)
dataset_train = SineDataset(n_x_train)
dataset_test = SineDataset(n_x_test)
# Now we need to pass the dataset to the Pytorch dataloader class along with some other arguments
# batch_size - the size of our mini-batches
# shuffle - whether or not we want to shuffle the dataset 
# For training shuffle is set to True and for Testing/Validation shuffle is set to False
data_loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
data_loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)


# In[9]:


set_background(red_bgd)

## Why is it neccessary for us to shuffle the training, and not shuffle validation/test?
# 普遍需求


# Now visualise the dataset you've created!!
# 
# Note: see here how we can just directly access the data from the dataset class. Make sure your generated sinewave matches the described sinewave within the dataset class!. 

# In[15]:


set_background(red_bgd)

# Visualize dataset using matplotlib scatter plot
train = torch.tensor([i for i in dataset_train])
test = torch.tensor([i for i in dataset_test])
fig1, (ax11,ax12) = plt.subplots(1,2)
ax11.scatter(train[...,0], train[...,1], s=1)
ax11.set_title("Train_data")
ax11.set_xlabel("x")
ax11.set_ylabel("y")

ax12.scatter(test[...,0], test[...,1], s=1)
ax12.set_title("Test_data")
ax12.set_xlabel("x")
ax12.set_ylabel("y")
plt.gcf().set_size_inches(20,10)
plt.show()


# ## Now, let's design a Neual Network and Train it!

# <h2> Neural Network Architecture</h2>
# 
# Up until now we have only created a single linear layer with an input layer and an output layer. In this Lab you will start to create multi-layered networks with many "hidden" layers separated by "activation functions" that give our networks "non-linearities". If we didn't have these activation functions and simple stacked layers together, our network would be no better than a single linear layer! Why? Because multiple sequential "linear transformations" can be modeled with just a single linear transformation. 
# 
# Neural networks are associated with a Directed Acyclic Graphs (DAG) describing how the functions are composed together. For example, we might have three functions $f^{(1)}, f^{(2)}$, and $f^{(3)}$, connected in a chain, to form in a chain, to form  $f(x) = f^{(3)}(f^{(2)}(f^{(1)}(x))).$ In this case, $f^{(1)}$ is called the first layer of the network, $f^{(2)}$ is called the second layer, and so on.  The ﬁnal layer of a feedforward network is called the output layer.
# 
# Except the output layer, the behavior of the other layers is not directly speciﬁed by the training data. The learning algorithm must decide how to use those layers to produce the desired output, but the training data do not say what each individual layer should do. Instead, the learning algorithm must decide how to use these layers to best implement an approximation of $f^*$. Because the training data does not show the desired output for each of these layers, and they
# are called hidden layers. These hidden layers, receive input from many other units and computes its own activation value. This requires us to choose the activation functions that will be used to compute the hidden layer values.
# 
# So what are these nonlinear activation functions that turn our simple linear models into a power "nonlinear function approximator"? Some common examples are:<br>
# 1. relu
# 2. sigmoid
# 3. tanh

# <h3>Pytorch Lightning Module</h3>
# 
# Now we can define a Pytorch Lightning model to be trained!<br>
# To do so, here you need to use the Pytorch LightningModule class as the base for defining your network. Just like the dataset class, this class has a number of important functions.
# A LightningModule is a PyTorch nn.Module and it has a few more helpful features.

# In[16]:


import os
import pandas as pd

from IPython.core.display import display
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from IPython.display import clear_output
from pytorch_lightning.callbacks import Callback, ModelCheckpoint


# ### 1.3 Design the Shallow Linear MLP model using PyTorch Lightning Module
# 
# Design a two layer Shallow MLP model and train it with both SGD and Adam optimizers. 

# In[30]:


set_background(red_bgd)

class ShallowLinear(LightningModule):
    def __init__(self, hidden_size=[1, 64, 1], learning_rate=1e-2, optimizer="SGD"):

        super().__init__()
        # Set our init args as class attributes, you can look at the PDF for the info
        self.learning_rate = learning_rate # Learning rate
        self.loss_fn = nn.MSELoss() # Use MSE loss as cost function
        #self.optimizer = optim.SGD(self.parameters(), lr=learning_rate) # Optimizer
        self.optimizer = optimizer
        
        D_in, H, D_out = hidden_size[0], hidden_size[1], hidden_size[2]
        
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        self.test_accuracy = Accuracy()
        
    def forward(self, x):
        # This function is an important one and we must create it or pytorch will give us an error!
        # This function defines the "forward pass" of our neural network
        x = x.view(x.size(0),1)
        #Lets define the sqeuence of events for our forward pass!
        x = self.linear1(x) # hidden layer
        x = torch.tanh(x) # activation function --> use tanh

        #No activation function on the output!!
        x = self.linear2(x) # output layer
        
        #Note we re-use the variable x as we don't care about overwriting it 
        #though in later labs we will want to use earlier hidden layers
        #later in our network!
        return x

    def training_step(self, batch, batch_idx):
        # Write training step
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits,y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Write validation step
        x, y = batch
        logits = self(x)
        #print(f"val: {x.shape, y.shape, logits.shape}")
        logits = logits.squeeze(1)
        loss = self.loss_fn(logits,y)

    def test_step(self, batch, batch_idx):
        # Write testing step
        x, y = batch
        logits = self(x)
        logits = logits.squeeze(1)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        # Write predict step
        x, y = batch
        return [x, self(x)]
        
    def configure_optimizers(self):
        # Return Adam or SGD optimizer
        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def train_dataloader(self):
        return data_loader_train
    
    def val_dataloader(self):
        return data_loader_test

    def test_dataloader(self):
        return data_loader_test


# ### 1.4 Define training and testing methods for models
# 
# By using the Trainer Constructor you can train and test the model.
# Also Trainer will automatically enables: 
# 1. Tensorboard logging 
# 2. Model checkpointing 
# 3. Training and validation loop 
# 4. early-stopping
# 
# In this task, we will test the model performance for both optimizers:
# - SGD
# - ADAM
# 

# In[18]:


set_background(red_bgd)

# Initialize Model with SGD optimizer
model_sgd = model_sgd = ShallowLinear()

# Train model for 20 epochs
trainer_task1_SGD = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=20,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=CSVLogger(save_dir="logs_task1_SGD/"),
)
trainer_task1_SGD.fit(model_sgd)

# Evaluate Model
trainer_task1_SGD.test()


# If your implementation is correct, you should obtain a printed output similar to this:
# 
# [{'test_loss': 0.009513414464890957}]

# In[31]:


set_background(red_bgd)

# Initialize Model with Adam optimizer
model_adam = ShallowLinear(optimizer="Adam")

# Train model for 20 epochs
trainer_task1_ADAM = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=20,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=CSVLogger(save_dir="logs_task1_Adam/"),
)
trainer_task1_ADAM.fit(model_adam)

# Evaluate Model
trainer_task1_ADAM.test()


# If your implementation is correct, you should obtain a printed output similar to this:
# 
# [{'test_loss': 0.005056165624409914}]

# ### 1.5 Generate scatter plots for above two trained models to compare noisy datapoints and denoised predictions

# In[26]:


set_background(red_bgd)

# Scatter plot for model trained using SGD (call predict function)
predictions_sgd = trainer_task1_SGD.predict(dataloaders=data_loader_test)
fig2, ax2 = plt.subplots(1,1)
ax2.scatter(test[...,0], test[...,1], s=1, c='g', label="Test_data")
for dat_a in predictions_sgd:
    ax2.scatter(dat_a[0], dat_a[1], s=1, c='r')
ax2.set_title("Predicted using SGD optimizer")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
plt.legend()
plt.show()


# In[27]:


set_background(red_bgd)

# Scatter plot for model trained using Adam (call predict function)
predictions_adam = trainer_task1_ADAM.predict(dataloaders=data_loader_test)
fig3, ax3 = plt.subplots(1,1)
ax3.scatter(test[...,0], test[...,1], s=1, c='g', label="Test_data")
for dat_a in predictions_adam:
    ax3.scatter(dat_a[0], dat_a[1], c="r", s=1)
ax3.set_title("Predicted using Adam optimizer")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
plt.legend()
plt.show()


# ### 1.6 Visualize experimental results
# 
# Pytorch-lightning module has its own built in methods to log values. These log files can be then used to visualize experimental results. You can use python plotting libraries such as matplotlib.
# 
# Using those logs, plot train losses for models trained on SGD and Adam.
# 
# Note: Make sure to drop NaN entries from dataframes before you plot using matplotlib.
# 
# You can read the logs by using the pandas library (pd.read_csv)

# In[28]:


set_background(red_bgd)

# read logs for model trained with SGD
training_loss_sgd = pd.read_csv(trainer_task1_SGD.logger.log_dir + "/metrics.csv")
training_loss_sgd = training_loss_sgd["train_loss"].dropna()

# read logs for model trained with Adam
training_loss_adam = pd.read_csv(trainer_task1_ADAM.logger.log_dir + "/metrics.csv")
training_loss_adam = training_loss_adam["train_loss"].dropna()


# In[29]:


set_background(red_bgd)

# Visualize train losses for model trained with SGD and Adam
fig4, (ax41, ax42) = plt.subplots(1,2)

ax41.plot(training_loss_sgd)
ax41.set_title("SGD Train_Loss", fontsize=18)
ax41.set_xticks(np.arange(len(training_loss_sgd)))
ax41.set_xlabel("Epoch")
ax41.set_ylabel("Loss")

ax42.plot(training_loss_adam)
ax42.set_title("Adam Train_Loss", fontsize=18)
ax42.set_xticks(np.arange(len(training_loss_sgd)))
ax42.set_xlabel("Epoch")
ax42.set_ylabel("Loss")

plt.gcf().set_size_inches(20,10)
plt.show()


# ### 1.7 Analysing above results, what optimizer you will choose? Explain why?

# In[32]:


set_background(red_bgd)

# Answer here
# 我会选择SGD优化器。它的损失更低，收敛性较好


# ### 1.8 Discuss why this simple shallow network is able to denoise noisy data and approximate sine function ?

# In[33]:


set_background(red_bgd)

# Answer here
#因为使用了tanh激活函数，增加神经网络模型的非线性


# ## Section 2 & 3 : Compare Shallow MLP and Deep MLP
# 
# 
# <div class="alert alert-block alert-info">
# 
# ## What you should do these sections 2 and 3!!
# 
# In sections 2 and 3 you will be training a Shallow MLPs and Deep MLPs for classifying images on the data file "section2_data.npz" which is the Fashion-MNIST dataset that contains images and labels for training, validation and test purposes (The images are 28x28 grayscale images).
# 
# You have to use Pytorch inbuilt datasets, Pytorch Lightning module class to construct a MLP in order to perform training and testing on the given datasets using stochastic gradient descent (SGD).
#     
# #### In you will work on the following points:
#  1. Create training, validation and testing dataloaders.
#  2. Visualize training datasets.
#  3. Design shallow neural network model.
#  4. Perform training the model and evaluation for different settings. Report test accuracies.
#  5. Visualize experimental results for training losses.
#  6. Visualize experimental results for validation accuracies.
#  7. Visualize predictions.
#  8. Optimize Shallow network's performance.
#     
#     
# #### In section 3 you will work on the following points:
#  1. Design deep neural network model.
#  2. Perform training the model and evaluation for different settings. Report test accuracies.
#  3. Visualize experimental results for training loss vs Validation accuracy.
#  4. Visualize predictions.
#  5. Discussion on experimental results.
# 
# <img src="figures/shallow_deep_mlp.png" width="1200" align="center">
#     
# In the above figure A is a shallow neural network and B is a deep neural network.
#     
# Note:  In all the parts below, you should only use the training images and their labels to train your model. You may use the **validation set** to pick a trained model. For example, during training, you can test the accuracy of your model using the validation set every epoch and pick the model that achieves the highest validation accuracy. You should then report your results on the test set once you choose your model.
# 

# ### 2.1 Create custom dataloader
# 
# The following class reads the data for section 2 & 3 tasks and creates a torch dataset object for it. With this, you can easily 
# use a dataloader to train your model. 
# 
# Note: Make sure that the file "section2_data.npz" is located properly (in this example, it should be in the same folder as this notebook).

# In[34]:


import torchvision

# For reproducability
torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 100 


# In[35]:


set_background(red_bgd)

class Section3Data(Dataset):
    def __init__(self, trn_val_tst = 0):
        # Load numpy data
        data = np.load("section2_data.npz")
        if trn_val_tst == 0:
            # Create dataset for trainloader --> images arr_0, labels arr_1
            self.images = data["arr_0"].T
            self.labels = data["arr_1"]
        elif trn_val_tst == 1:
            # Create dataset for valloader --> images arr_2, labels arr_3
            self.images = data["arr_2"].T
            self.labels = data["arr_3"]
        else:
            # Create dataset for testloader --> images arr_4, labels arr_5
            self.images = data["arr_4"].T
            self.labels = data["arr_5"]
            
        # Normalize images [0,1]  
        self.images = (self.images - self.images.min())/(self.images.max() - self.images.min())

    # Define len function
    def __len__(self):
        return len(self.labels)

    # Define getitem function
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
   
        sample = self.images[idx,:]
        labels = self.labels[idx]
        return sample, labels


# Now using dataset class create a dataloader for the section 3 data.

# In[36]:


set_background(red_bgd)

# Call training dataset and create the trainloader
trainset = Section3Data(trn_val_tst=0)
trainloader = DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True)

# Call validation dataset and create the trainloader
valset = Section3Data(trn_val_tst=1)
valloader = DataLoader(valset,batch_size=BATCH_SIZE,shuffle=False)

# Call testing dataset and create the trainloader
testset = Section3Data(trn_val_tst=2)  
testloader = DataLoader(testset,batch_size=BATCH_SIZE,shuffle=False)


# ### 2.2 Visualize dataset
# 
# Lets visualise that mini-batches that the training dataloader gives us.

# In[38]:


set_background(red_bgd)


# We can create an iterater using the dataloaders and take a random sample 
train_data_iter = iter(trainloader)
data, labels = train_data_iter.next()
data = data.reshape(100, 1, 28, -1)
print(f"data:\t{data.shape}")
print(f"label:\t{labels.shape}")

# visualize dataset using torchvision grid
out = torchvision.utils.make_grid(data, nrow=16)

fig5, ax5 = plt.subplots(1,1)
ax5.imshow(out.numpy().transpose((1, 2, 0)))
ax5.set_title("Fashion_MNIST Dataset", fontsize=20)
plt.gcf().set_size_inches(20,10)
plt.show()


# ### 2.3 Design Shallow MLP
# 
# Design a Shallow MLP with one hidden linear layer.

# In[39]:


set_background(red_bgd)

class Shallow_MLP(LightningModule):
    def __init__(self, n, learning_rate=1e-1):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.loss_fun = nn.CrossEntropyLoss()  # Use CE loss
        
        self.linear1 =  nn.Linear(784, n)
        self.linear2 =  nn.Linear(n, 10)
        
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
            
    def forward(self, x):
        #Pass input through conv layers
        bs = x.shape[0]
        x = x.reshape(bs, -1).float()
        out1 = torch.tanh(self.linear1(x))
        out2 = self.linear2(out1)
    
        return out2
    
    def training_step(self, batch, batch_idx):
        # Write training step
        x, y = batch
        logits = self(x)
        loss = self.loss_fun(logits, y)

        preds = logits.argmax(1)
        self.train_accuracy.update(preds, y)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx):
        # Write validation step
        x, y = batch
        logits = self(x)
        loss = self.loss_fun(logits, y)

        preds = logits.argmax(1)
        self.val_accuracy.update(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # Write testing step
        x, y = batch
        logits = self(x)
        loss = self.loss_fun(logits, y)
        
        preds = logits.argmax(1)
        self.test_accuracy.update(preds, y)
        
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
    def predict_step(self, batch, batch_idx):
        # Write predict step
        x, y = batch
        logits = self(x)
        preds = logits.argmax(1)
        return [preds, y]
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):
        return trainloader
    
    def val_dataloader(self):
        return valloader

    def test_dataloader(self):
        return testloader


# ### 2.4 Train and evaluate model's performance for following scenarios
# 
# a. fc1 : Linear(784 $\times$ 32) $\rightarrow$ ReLU $\rightarrow$ fc2 : Linear(32 x$\times$ 10)
# 
# b. fc1 : Linear(784 $\times$ 128) $\rightarrow$ ReLU $\rightarrow$ fc2 : Linear(128 $\times$ 10)
# 
# c. fc1 : Linear(784 $\times$ 512) $\rightarrow$ ReLU $\rightarrow$ fc2 : Linear(512 $\times$ 10)
# 
# In all the parts above, you should only use the training images and their labels to train your model. You may use the validation set to pick a trained model.
# 
# For example, during training, you can test the accuracy of your model using the validation set every epoch and pick the model that achieves the highest validation accuracy. You should then report your results on the test set once you choose your model. 
# 
# Note: Make sure to have different log file directories and checkpoint folders (eg: logs_task_2a, logs_task_2b and logs_task_2c)

# In[40]:


set_background(red_bgd)

# Initialize Shallow MLP model with n = 32
model = Shallow_MLP(n=32)

# Define checkpoint callback function to save best model
checkpoint_callback_2a = ModelCheckpoint(monitor="val_acc",dirpath="logs_task_2a/",save_top_k=1,mode="max",every_n_epochs=1)

# Train and test the model
trainer_2a = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  
    max_epochs=100,
    callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback_2a],
    logger=CSVLogger(save_dir="logs_task_2a/"),
)
trainer_2a.fit(model)
trainer_2a.test()


# If your implementation is correct, you should obtain a printed output similar to this:
# 
# [{'test_loss': 0.4268353581428528, 'test_acc': 0.8521000146865845}]

# In[41]:


set_background(red_bgd)

# Initialize Shallow MLP model with n = 128
model = Shallow_MLP(n=128)

# Define checkpoint callback function to save best model
checkpoint_callback_2b = ModelCheckpoint(monitor="val_acc",dirpath="logs_task_2b/",save_top_k=1,mode="max",every_n_epochs=1)

# Train and test the model
trainer_2b = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  
    max_epochs=100,
    callbacks=[TQDMProgressBar(refresh_rate=20),checkpoint_callback_2b],
    logger=CSVLogger(save_dir="logs_task_2b/"),
)
trainer_2b.fit(model)
trainer_2b.test()


# If your implementation is correct, you should obtain a printed output similar to this:
#     
# [{'test_loss': 0.4281507730484009, 'test_acc': 0.8611000180244446}]

# In[42]:


set_background(red_bgd)

# Initialize Shallow MLP model with n = 512
model = Shallow_MLP(n=512)

# Define checkpoint callback function to save best model
checkpoint_callback_2c = ModelCheckpoint(monitor="val_acc",dirpath="logs_task_2c/",save_top_k=1,mode="max",every_n_epochs=1)

# Train and test the model
trainer_2c = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None, 
    max_epochs=100,
    callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback_2c],
    logger=CSVLogger(save_dir="logs_task_2c/"),
)
trainer_2c.fit(model)
trainer_2c.test()


# If your implementation is correct, you should obtain a printed output similar to this:
#     
# [{'test_loss': 0.4100257456302643, 'test_acc': 0.8678666949272156}]

# ### 2.5 Plot Training losses for the different shallow networks

# In[43]:


set_background(red_bgd)

# read logs for 2a
training_data_2a = pd.read_csv(trainer_2a.logger.log_dir + "/metrics.csv")
training_loss_2a = training_data_2a["train_loss"].dropna()
# read logs for 2b
training_data_2b = pd.read_csv(trainer_2b.logger.log_dir + "/metrics.csv")
training_loss_2b = training_data_2b["train_loss"].dropna()
# read logs for 2c
training_data_2c = pd.read_csv(trainer_2c.logger.log_dir + "/metrics.csv")
training_loss_2c = training_data_2c["train_loss"].dropna()


# In[45]:


set_background(red_bgd)

# Plot using matplotlib
fig6, ax6 = plt.subplots(1,1)
ax6.plot(training_loss_2a, label="n = 32")
ax6.plot(training_loss_2b, label="n = 128")
ax6.plot(training_loss_2c, label="n = 512")
ax6.set_title("The train_loss of Shallow MLP ")
ax6.set_xlabel("Epochs")
ax6.set_ylabel("Loss")
plt.legend()
plt.show()
# Discuss what you see


# ### 2.6 Plot Validation accuracies for the different shallow networks

# In[46]:


set_background(red_bgd)

# Plot using matplotlib
val_acc_2a = training_data_2a["val_acc"].dropna()
val_acc_2b = training_data_2b["val_acc"].dropna()
val_acc_2c = training_data_2c["val_acc"].dropna()
fig7, ax7 = plt.subplots(1,1)
ax7.plot(val_acc_2a, label="n = 32")
ax7.plot(val_acc_2b, label="n = 128")
ax7.plot(val_acc_2c, label="n = 512")
ax7.set_title("The Val_acc of Shallow MLP")
ax7.set_xlabel("Epochs")
ax7.set_ylabel("Accuracy")
plt.legend()
plt.show()


# ### 2.7 Visualize 5 predictions of test set along with groundtruths and input images for 3rd Shallow MLP

# In[49]:


set_background(red_bgd)

# Generate predictions using predict function
predictions_2c = trainer_2c.predict(dataloaders=testloader)

# visualize predictions along with ground truths and input images using matplotlib
test_loader_iter = iter(testloader)
test_data, test_labels = test_loader_iter.next()
def vis_preds(testloader, predictions):
    test_data, test_labels = next(iter(testloader))
    im_idx = torch.randint(0, test_labels.shape[0], (5,))
    im_label = [predictions[0][0][i] for i in im_idx]
    im_pred = [predictions[0][1][i] for i in im_idx]
    fig, axs = plt.subplots(1, 5, figsize=(18,6), sharey=True)
    for i, im_num in enumerate(im_idx):
        axs[i].imshow(test_data[im_num].resize(28,28))
        axs[i].set_title(f"Item {im_num}: label: {im_label[i]}, pred: {im_pred[i]}")
    fig.suptitle("Test_pred Visual", fontsize=15)
    plt.show()
    
vis_preds(testloader, predictions_2c)


# ### 2.8 Optimize hyper-parameters to improve accuracy of shallow network
# 
# What do you recommend to improve shallow network's accuracy further ?
# 
# - Make sure to test at least 2-3 hyper parameters
# - For each hyper parameter, have 3 variations
# - Plot out the results after you train + tested it. It is recommended to use subplot (or overlay onto 1 plot the accuracies for each model). **Important! Think about which accuracy is most relevant** 

# In[50]:


set_background(red_bgd)

# What do you recommend?
# 对于学习速率，默认的 1e-1 在训练损失和验证准确性方面表现最佳，这更为重要。这是因为验证集是模型尚未看到的新数据，因此它是对模型的普遍性的真实测试。我们看到，降低学习速率并不能取得多大成就，如果有的话，也会损害模型的准确性，也会减慢其收敛速度。。


# Train and test the Shallow MLP model using above suggested. When testing different hyper-parameters, you can use the original shallow MLP class from earlier, and just train using the different hyper-parameters. Save your MLP with the optimized parameters. 

# In[51]:


set_background(red_bgd)

# Create New Class
# class Shallow_MLP_Optimized(LightningModule):
set_background(red_bgd)

class Shallow_MLP_Optimised(LightningModule):
    def __init__(self, learning_rate=1e-1):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.loss_fun = nn.CrossEntropyLoss() # Use CE loss
        
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 10)
        
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
            
    def forward(self, x):
        #Pass input through conv layers
        bs = x.shape[0]
        x = x.reshape(bs, -1).float()
        out1 = torch.tanh(self.linear1(x))
        out2 = self.linear2(out1)
    
        return out2
    
    def training_step(self, batch, batch_idx):
        # Write training step
        x, y = batch
        logits = self(x)
        loss = self.loss_fun(logits, y)

        preds = logits.argmax(1)
        self.train_accuracy.update(preds, y)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Write validation step
        x, y = batch
        logits = self(x)
        loss = self.loss_fun(logits, y)

        preds = logits.argmax(1)
        self.val_accuracy.update(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # Write testing step
        x, y = batch
        logits = self(x)
        loss = self.loss_fun(logits, y)
        
        preds = logits.argmax(1)
        self.test_accuracy.update(preds, y)
        
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
    def predict_step(self, batch, batch_idx):
        # Write predict step
        x, y = batch
        logits = self(x)
        preds = logits.argmax(1)
        return [preds, y]

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):
        return trainloader
    
    def val_dataloader(self):
        return valloader

    def test_dataloader(self):
        return testloader
# Train the model
model = Shallow_MLP_Optimised()

checkpoint_callback_opt_final = ModelCheckpoint(
    monitor="val_acc",
    dirpath="logs_task_2_opt_final/",
    save_top_k=1,
    mode="max",
    every_n_epochs=1
)

trainer_opt_final = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  
    max_epochs=150,
    callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback_opt_final],
    logger=CSVLogger(save_dir="logs_task_2_opt_final/"),
)
trainer_opt_final.fit(model)
# Test the model
trainer_opt_final.test()
# Visualize accuracy graphs and compare it with 2.3 Accuracies
training_data_opt_final = pd.read_csv(trainer_opt_final.logger.log_dir + "/metrics.csv")
training_loss_opt_final = training_data_opt_final["train_loss"].dropna()
val_acc_opt_final = training_data_opt_final["val_acc"].dropna()

fig10, (ax101, ax102) = plt.subplots(1, 2)
ax101.plot(training_loss_2b, label="origin")
ax101.plot(training_loss_opt_final, label="optimize")
ax101.set_title("Train_Loss", fontsize=20)
ax101.set_xlabel("Epochs")
ax101.set_ylabel("Loss")
ax101.legend(prop={'size': 15})

ax102.plot(val_acc_2b, label="original")
ax102.plot(val_acc_opt_final, label="optimize")
ax102.set_title("Val_acc", fontsize=20)
ax102.set_xlabel("Epochs")
ax102.set_ylabel("Accuracy")
ax102.legend(prop={'size': 15})

plt.gcf().set_size_inches(20,10)
plt.show()


# # Section 3: Deep MLP

# ### 3.1 Design Deep MLP
# 
# Design a Deep MLP with four linear layers.

# In[52]:


set_background(red_bgd)


class Deep_MLP(LightningModule):
    def __init__(self, n1, n2, n3, learning_rate=1e-1):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.loss_fun = nn.CrossEntropyLoss() # Use CE loss
        
        self.linear1 = nn.Linear(784, n1)
        self.linear2 = nn.Linear(n1, n2)
        self.linear3 = nn.Linear(n2, n3)
        self.linear4 = nn.Linear(n3, 10)
    
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
            
    def forward(self, x):
        #Pass input through conv layers
        
        bs = x.shape[0]
        x = x.reshape(bs, -1).float()
        out1 = torch.tanh(self.linear1(x))
        out2 = torch.tanh(self.linear2(out1))
        out3 = torch.tanh(self.linear3(out2))
        out4 = self.linear4(out3)
      
        return out4
    
    def training_step(self, batch, batch_idx):
        # Write training step
        x, y = batch
        logits = self(x)
        loss = self.loss_fun(logits, y)
        preds = logits.argmax(1)
        self.train_accuracy.update(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)        
        return loss
    def validation_step(self, batch, batch_idx):
        # Write validation step
        x, y = batch
        logits = self(x)
        loss = self.loss_fun(logits, y)
        preds = logits.argmax(1)
        self.val_accuracy.update(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True, on_step=False, on_epoch=True)
    def test_step(self, batch, batch_idx):
        # Write testing step
        x, y = batch
        logits = self(x)
        loss = self.loss_fun(logits, y)
        preds = logits.argmax(1)
        self.test_accuracy.update(preds, y)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True, on_step=False, on_epoch=True)
    def predict_step(self, batch, batch_idx):
        # Write predict step
        x, y = batch
        logits = self(x)
        preds = logits.argmax(1)
        return [preds, y]

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):
        return trainloader
    
    def val_dataloader(self):
        return valloader

    def test_dataloader(self):
        return testloader


# ### 3.2 Train and evaluate model's performance for following scenarios
# 
# a. fc1 : Linear(784 $\times$ 128) $\rightarrow$ ReLU $\rightarrow$ fc2 : Linear(128 $\times$ 64) $\rightarrow$ ReLU $\rightarrow$ fc3 : Linear(64 $\times$ 32) ->$\rightarrow$ Relu $\rightarrow$ fc4 : Linear(32 $\times$ 10)
# 
# b. fc1 : Linear(784 $\times$ 32) $\rightarrow$ ReLU $\rightarrow$ fc2 : Linear(32 $\times$ 64) $\rightarrow$ ReLU $\rightarrow$ fc3 : Linear(64 $\times$ 128) $\rightarrow$ Relu $\rightarrow$ fc4 : Linear(128 $\times$ 10)
# 
# c. fc1 : Linear(784 $\times$ 64) $\rightarrow$ ReLU $\rightarrow$ fc2 : Linear(64 $\times$ 64) $\rightarrow$ ReLU $\rightarrow$ fc3 : Linear(64 $\times$ 64) $\rightarrow$ Relu $\rightarrow$ fc4 : Linear(64 $\times$ 10)
# 
# Note: Make sure to have different log file directories (eg: logs_task_3a, logs_task_3b and logs_task_3c)

# In[53]:


set_background(red_bgd)


# Initialize Deep MLP model for scenario 3a
model = Deep_MLP(n1=128, n2=64, n3=32)

# Define checkpoint callback function to save best model
checkpoint_callback_3a = ModelCheckpoint(monitor="val_acc",dirpath="logs_task_3a/",save_top_k=1,mode="max",every_n_epochs=1)

# Train and Test Model
trainer_3a = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  
    max_epochs=100,
    callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback_3a],
    logger=CSVLogger(save_dir="logs_task_3a/"),
)
trainer_3a.fit(model)
trainer_3a.test()


# If your implementation is correct, you should obtain a printed output similar to this:
# 
# [{'test_loss': 0.4924968481063843, 'test_acc': 0.8584499955177307}]

# In[54]:


set_background(red_bgd)

# Initialize Deep MLP model for scenario 3b
model = Deep_MLP(n1=32, n2=64, n3=128)

# Define checkpoint callback function to save best model
checkpoint_callback_3b = ModelCheckpoint(monitor="val_acc",dirpath="logs_task_3b/",save_top_k=1,mode="max",every_n_epochs=1)

# Train and Test Model
trainer_3b = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  
    max_epochs=100,
    callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback_3b],
    logger=CSVLogger(save_dir="logs_task_3b/"),
)
trainer_3b.fit(model)
trainer_3b.test()


# If your implementation is correct, you should obtain a printed output similar to this:
#     
# [{'test_loss': 0.4536038041114807, 'test_acc': 0.8570333123207092}]

# In[55]:


# Initialize Deep MLP model for scenario 3c
model = Deep_MLP(n1=64, n2=64, n3=64)

# Define checkpoint callback function to save best model
checkpoint_callback_3c = ModelCheckpoint(monitor="val_acc",dirpath="logs_task_3c/",save_top_k=1,mode="max",every_n_epochs=1)

# Train and Test Model
trainer_3c = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=100,
    callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback_3c],
    logger=CSVLogger(save_dir="logs_task_3c/"),
)
trainer_3c.fit(model)
trainer_3c.test()


# If your implementation is correct, you should obtain a printed output similar to this:
#     
# [{'test_loss': 0.40754958987236023, 'test_acc': 0.8532999753952026}]

# ### 3.3 Plot Training loss vs Validation loss for the different deep networks

# In[58]:


set_background(red_bgd)
# read logs for 3a
training_data_3a = pd.read_csv(trainer_3a.logger.log_dir + "/metrics.csv")
training_loss_3a = training_data_3a["train_loss"].dropna()
val_loss_3a = training_data_3a["val_loss"].dropna()
# read logs for 3b
training_data_3b = pd.read_csv(trainer_3b.logger.log_dir + "/metrics.csv")
training_loss_3b = training_data_3b["train_loss"].dropna()
val_loss_3b = training_data_3b["val_loss"].dropna()
# read logs for 3c
training_data_3c = pd.read_csv(trainer_3c.logger.log_dir + "/metrics.csv")
training_loss_3c = training_data_3c["train_loss"].dropna()
val_loss_3c = training_data_3c["val_loss"].dropna()


# In[59]:


set_background(red_bgd)

# Plot using matplotlib, you can overlay the graphs.
fig11, (ax111, ax112) = plt.subplots(1,2)
ax111.plot(training_loss_3a, label="n1 = 128, n2 = 64, n3 = 32")
ax111.plot(training_loss_3b, label="n1 = 32, n2 = 64, n3 = 128")
ax111.plot(training_loss_3c, label="n1 = 64, n2 = 64, n3 = 64")
ax111.set_title("The Train_Loss of Deep MLP", fontsize=20)
ax111.set_xlabel("Epochs")
ax111.set_ylabel("Loss")
ax111.legend(prop={'size': 15})
ax112.plot(val_loss_3a, label="n1=128, n2=64, n3=32")
ax112.plot(val_loss_3b, label="n1=32, n2=64, n3=128")
ax112.plot(val_loss_3c, label="n1=64, n2=64, n3=64")
ax112.set_title("The Val_Loss of Deep MLP", fontsize=20)
ax112.set_xlabel("Epochs")
ax112.set_ylabel("Loss")
ax112.legend(prop={'size': 15})
plt.gcf().set_size_inches(20,10)
plt.show()


# ### 3.4 Visualize 5 predictions of test set along with groundtruths and input images for 3rd Deep MLP

# In[60]:


set_background(red_bgd)

# Generate predictions using predict function
predictions_3c = trainer_3c.predict(dataloaders=testloader)

# visualize predictions along with ground truths and input images using matplotlib
test_loader_iter = iter(testloader)
test_data, test_labels = test_loader_iter.next()
vis_preds(testloader, predictions_3c)


# ### 3.5 Discussion:

# In[ ]:


set_background(red_bgd)

# Question: Did you observe a strange behaviour for the deep MLPs? If yes, what is that ? Discuss what might be the source of that.

# Answer here
# >> Validation loss seems to increase after around 75 epochs regardless of hidden layer sizes. This may be
# due to the increased complexity of the deep networks compared to shallow networks, forcing the model to 
# overfit to the training data. This is evident as the training loss seems to be decreasing at a comfortable
# rate, but this is not the case for validation.


# In[ ]:


set_background(red_bgd)

# Question: How does the deep MLP compare with the shallow MLP for this case?

# Answer here
# >> Performance seems to be quite similar, however there is the risk that the deep MLP has overfit. 
# Regardless of this, both validation accuracies seem to be pretty similar, around 84%.


# # Do not remove or edit the following code snippet. 
# 
# When submitting your report, please ensure that you have run the entire notebook from top to bottom. You can do this by clicking "Kernel" and "Restart Kernel and Run All Cells". Make sure the last cell (below) has also been run. 

# In[ ]:


file_name = str(student_number) + 'Lab_3_Student'
cmd = "jupyter nbconvert --to script Lab_3_Student.ipynb --output " + file_name
if(os.system(cmd)):
    print("Error converting to .py")
    print("cmd")


# <div class="alert alert-block alert-warning">
#     
# # <center> That is all for this Lab!
#     
# </div>
