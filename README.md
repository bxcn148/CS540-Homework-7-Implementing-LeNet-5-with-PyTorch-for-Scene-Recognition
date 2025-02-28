Download link :https://programming.engineering/product/cs540-homework-7-implementing-lenet-5-with-pytorch-for-scene-recognition/

# CS540-Homework-7-Implementing-LeNet-5-with-PyTorch-for-Scene-Recognition
CS540 Homework 7 Implementing LeNet-5 with PyTorch for Scene Recognition
Assignment Goals

Implement and train a convolutional neural network (CNN), specifically LeNet

Understand and count the number of trainable parameters in CNN

Explore different training configurations such as batch size, learning rate and training epochs.

Design and customize your own deep network for scene recognition

Summary

Your implementation in this assignment might take one or two hours to run. We highly recommend to start working on this assignment early! In this homework, we will explore building deep neural networks, particu larly Convolutional Neural Networks (CNNs), using PyTorch. Helper code is provided in this assignment.

In this HW, you are still expected to use the Pytorch library and virtual environment for programming. You can find the relevant tutorials in HW6 Part I. We omit here to avoid redundancy.

Design a CNN model for MiniPlaces Dataset

In this part, you will design a simple CNN model along with your own convolutional network for a more realistic dataset – MiniPlaces, again using PyTorch.

Dataset

MiniPlaces is a scene recognition dataset developed by MIT. This dataset has 120K images from 100 scene cate gories. The categories are mutually exclusive. The dataset is split into 100K images for training, 10K images for validation, and 10K for testing.


Figure 1: Examples of images in miniplaces

Our data loader will try to download the full dataset the first time you run train_miniplaces.py, but this may not work if the miniplaces website is down. Follow the instructions in section “Downloading miniplaces manually” to setup miniplaces manually.

Downloading from miniplaces website:

Run the command below. It may fail the first time you run it because your LeNet() is still empty, or expired SSL certificates, but the data will be unpacked in the correct format.

python3 train_miniplaces.py

Move train.txt and val.txt into the data/miniplaces folder so it should contain test, train, and val folders and train and val txt files. Miniplaces should now be ready to use

Downloading miniplaces manually:

Create an empty data folder (with name data) in your hw7 directory.

Download the backup miniplaces data manually and put in in data/

Go to data/

cd data/

4. Untar your downloaded data.tar.gz

tar -xvf data.tar.gz

The original tarball has `images` and `objects` directories. We only need `images`, you can remove `objects`

Rename the `images` folder as `miniplaces`

mv images/ miniplaces/

Move train.txt and val.txt into the data/miniplaces folder so it should contain test, train, and val folders and train and val txt files. Miniplaces should now be ready to use

In train_miniplaces.py, modify to download=False in lines 59 and 61.

If you are using CSL machine, you can scp the downloaded tarball for use in CSL:

scp data.tar.gz <userid>@best-linux.cs.wisc.edu:~

Helper Code

We provide helper functions in train_miniplaces.py and dataloader.py, and skeleton code in student_code.py.

See the comments in these files for more implementation details.

The original image resolution for images in MiniPlaces is 128×128. To make the training feasible, our data loader reduces the image resolution to 32×32. You can always assume this input resolution.

Before the training procedure, we define the dataloader, model, optimizer, image transform and criterion. We execute the training and testing in function train_model and test_model, which is similar to what we have for HW6.

Part I Creating LeNet-5

Background: LeNet was one of the first CNNs and its success was foundational for further research into deep learning. We are implementing an existing architecture in the assignment, but it might be interesting to think about why early researchers chose certain kernel sizes, padding, and strides which we learned about conceptually in class.

In this part, you have to implement a classic CNN model, called LeNet-5 in Pytorch for the MiniPlaces dataset.

We use the following layers in this order:

One convolutional layer with the number of output channels to be 6, kernel size to be 5, stride to be 1, followed by a relu activation layer and then a 2D max pooling layer (kernel size to be 2 and stride to be 2).

One convolutional layer with the number of output channels to be 16, kernel size to be 5, stride to be 1, followed by a relu activation layer and then a 2D max pooling layer (kernel size to be 2 and stride to be 2).

A Flatten layer to convert the 3D tensor to a 1D tensor.

A Linear layer with output dimension to be 256, followed by a relu activation function.

A Linear layer with output dimension to be 128, followed by a relu activation function.

A Linear layer with output dimension to be the number of classes (in our case, 100).

You have to fill in the LeNet class in student_code.py. You are expected to create the model following this tutorial, which is different from using nn.Sequential() in the last HW.

In addition, given a batch of inputs with shape [N, C, W, H], where N is the batch size, C is the input channel and W, H are the width and height of the image (both 32 in our case), you are expected to return both the output of the model along with the shape of the intermediate outputs for the above 6 stages. The shape should be a dictionary with the keys to be 1,2,3,4,5,6 (integers) denoting each stage, and the corresponding value to be a list

that denotes the shape of the intermediate outputs.

To help visualization, recall the LeNet architecture from lecture:


Figure 2: LeNet Architecture

Hints

The expected model has the following form:

class LeNet(nn.Module):

def __init__(self, input_shape=(32, 32), num_classes=100): super(LeNet, self).__init__()

certain definitions

def forward(self, x): shape_dict = {}

certain operations return out, shape_dict

Shape dict should has the following form:

{1: [a,b,c,d], 2:[e,f,g,h], .., 6: [x,y]}

The linear layer and the convolutional layer have bias terms.

You should need to use the conv2d function to create a convolutional layer. The parameters allow us to specify the details of the layer eg. the input/output dimensions, padding, stride, and kernel size. More information can be found in the documentation.

Part II Count the number of trainable parameters of LeNet-5

Background: As discussed in lecture, fully connected models (like what we made in the previous homework) are dense with many parameters we need to train. After finishing this part, it might be helpful to think about the number of parameters in this model compared to the number of parameters in a fully connected model of similar depth (similar number of layers). Especially, how does the difference in size impact efficiency and accuracy?

In this part, you are expected to return the number of trainable parameters of your created LeNet model in the previous part. You have to fill in the function count_model_params in student_code.py.

The function output should be in the unit of Million (1e6). This means if your model has 5.325 * 1e6 parameters, your function should return 5.325. Please do not use any external library which directly calculates the number of parameters (other libraries, such as NumPy can be used as helpers)!

Hint: You can use the model.named_parameters() to gain the name and the corresponding parameters of a model. Please do not do any rounding to the result.

Part III Training LeNet-5 under different configurations

Background: A large part of creating neural networks is designing the architecture (part 1). However, there are other ways of tuning the neural net to change its performance. In this section, we can see how batch size, learning rate, and number of epochs impact how well the model learns. As you get your results, it might be helpful to think about how and why the changes have impacted the training.

Based on the LeNet-5 model created in the previous parts, in this section, you are expected to train the LeNet-5 model under different configurations.

You will use similar implementations of train_model and test_model as you did for HW6 (which we provide in student_code.py). When you run train_miniplaces.py, the python script will save two files in the ”outputs” folder.

checkpoint.pth.tar is the model checkpoint at the latest epoch.

model_best.pth.tar is the model weights that has highest accuracy on the validation set.

Our code supports resuming from a previous checkpoint, such that you can pause the training and resume later. This can be achieved by running

python train_miniplaces.py –resume ./outputs/checkpoint.pth.tar

After training, we provide eval_miniplaces.py to help you evaluate the model on the validation set and also help you in timing your model. This script will grab a pre-trained model and evaluate it on the validation set of 10K images. For example, you can run

python eval_miniplaces.py –load ./outputs/model_best.pth.tar

The output shows the validation accuracy and also the model evaluation time in seconds (see an example below).

=> Loading from cached file ./data/miniplaces/cached_val.pkl … => loading checkpoint ’./outputs/model_best.pth.tar’

=> loaded checkpoint ’./outputs/model_best.pth.tar’ (epoch x)

Evaluting the model …

[Test set] Epoch: xx, Accuracy: xx.xx%

Evaluation took 2.26 sec

You can run this script a few times to see the average runtime of your model.

Please train the model under the following different configurations:

The default configuration provided in the code, which means you do not have to make modifications.

Set the batch size to 8, the remaining configurations are kept the same.

Set the batch size to 16, the remaining configurations are kept the same.

Set the learning rate to 0.05, the remaining configurations are kept the same.

Set the learning rate to 0.01, the remaining configurations are kept the same.

Set the epochs to 20, the remaining configurations are kept the same.

Set the epochs to 5, the remaining configurations are kept the same.

After training, you are expected to get the validation accuracy as stated above using the best model (model_best.pth.tar), then save these accuracies into a results.txt file (you can do this manually), where the accuracy of each configuration is placed in one line in order. Your .txt file will end up looking like this:

18.00

17.39

12.31

…

The exact accuracy may not align well with your results. They are just for illustration purposes. You have to submit the results.txt file together with your student_code.py.

Optional: Profiling Your Model

You might find that the training or evaluation of your model is a bit slower than expected. Fortunately, PyTorch has its own profiling tool. Here is a quick tutorial of using PyTorch profiler. You can easily inject the profiler into train_miniplaces.py to inspect the runtime and memory consumption of different parts of your model. A general principle is that a deep (many layers) and wide (many feature channels) network will train much slower. It is your design choice to balance between the efficiency and the accuracy.

Optional: Training on CSL

We recommend training the model on your local machine and time your model (using eval_miniplaces.py) on CSL machines. If you decide to train the model on a CSL machine, you will need to find a way to allow your remote session to remain active when you are disconnected from CSL. In this case, we recommend using tmux, a terminal multiplexer for Unix-like systems. tmux is already installed on CSL. To use tmux, simply type tmux in the terminal. Now you can run your code in a tmux session. And the session will remain active even if you are disconnected.

If you want to detach a tmux session without closing it, press ”ctrl + b” then ”d” (detach) within a tmux session. This will exist to the terminal while keeping the session active. Later you can re-attach the session.

If you want to enter an active tmux session, type ”tmux a” to attach to the last session in the terminal (outside of tmux).

If you want to close a tmux session, press ”ctrl + b” then ”x” (exit) within a tmux session. You won’t be able to enter this session again. Please make sure that you close your tmux sessions after this assignment.

See here or here for a brief tutorial on the powerful tool of tmux.

Deliverables

You will need to submit student_code.py together with your results.txt for Part III. The validation accuracy for different configurations must be in .txt format and named as results.txt

Submission Notes

You will need to submit a single zip file named hw7_<netid>.zip, where you replace <netid> with your netID (your wisc.edu login). This file will include student_code.py and the validation accuracy file with the proper name as mentioned above. Do not submit a Jupyter notebook .ipynb file. Be sure to remove all debugging output before submission. Failure to remove debugging output will be penalized. This assignment is due on April 11. We highly recommend to start early.

In this assignment, we give you sample_autograder.py to test superficial aspects of your code (input dimension, number of outputs expected, etc). Note that getting 100 from this autograder does not mean that you will get 100 in your final grade, it means that you have the correct input output types.

Common Errors

When miniplaces website is down, you may get urllib.error.HTTPError: HTTP Error 404: Not Found

If you have the data downloaded already, you can change download=False whenever you instantiate MiniPlaces() class

