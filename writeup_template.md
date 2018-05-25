# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of images under each unique class for each of train,test and validation data set.
Train dataset
![alt text](/home/anooj/mdfileimages/index1.png)
Test dataset
![alt text](/home/anooj/mdfileimages/index2.png)
Validation data set
![alt text](/home/anooj/mdfileimages/index1.png)


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

This is done in code cells number 5,6 and 7. As a first step, I decided to convert the images to grayscale because being a single channel image reduces the complexity for the model to identify an image. 
Here is an example of a traffic sign image before and after grayscaling.
![alt text][/home/anooj/mdfileimages/clr.png]


![alt text][/home/anooj/mdfileimages/bw.png]

As a last step, I normalized the image data because it aides in speed training and efficient utilization of the available resources.


Here is an example of normalized image
![alt text][/home/anooj/mdfileimages/nr.png]




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:
*************************************************
Layer               	Description
Input               	32x32x1 grayscale image
Convolution 5x5     	2x2 stride, valid padding, outputs 28x28x6
RELU 	
Max pooling         	2x2 stride, outputs 14x14x6
Convolution 5x5     	2x2 stride, valid padding, outputs 10x10x16
RELU 	
Max pooling         	2x2 stride, outputs 5x5x16
Convolution 1x1      	2x2 stride, valid padding, outputs 1x1x412
RELU 	
Fully connected     	input 412, output 122
RELU 	
Dropout             	50% keep
Fully connected     	input 122, output 84
RELU 	
Dropout             	50% keep
Fully connected     	input 84, output 43





#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 9th 10th and 11th cell of the ipython notebook. To train the model, I used LeNet  but with an additional convolution without a max pooling layer after it. I used the AdamOptimizer with a learning rate of 0.001. The epochs used was 12 while the batch size was 128.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of    99.1%
* validation set accuracy of  97.7% 
* test set accuracy of        91.4% 

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 
     The same architecture as demonstrated in the Udacity class as it got a good score.
* What were some problems with the initial architecture?
     The Lenet model worked well but the validation accuracy was mostly in the 70-75 percentage range. Grey scaling and normalization of the image increased the accuracy and addition of another convolution layer to the architecture finaly helped me reach higher accuracy levels of around 98%
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
      An additional convolution layer and a couple of dropout 50 % was added to the LeNet architecture.
* Which parameters were tuned? How were they adjusted and why?
      Parameters such as Epoch, learning rate, batch size were tuned. Learning rate was finally set to a standard .001 after I experimented with several smaller and slightly larger values for the same. Batch size was tested with slightly higher values in the range of 150s initially and then lowered from their till i reached the best accuracy levels. Selected 12 Epochs as the model showed tendancy of converging around 10 epochs and wanted to train for another couple of epochs so that I can be sure it has reached the maximum accuracy it can.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  Adding another convolution layer was the important design choice made which greatly affected the output accuracy of the model.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][/home/anooj/mdfileimages/1.png] ![alt text][/home/anooj/mdfileimages/2.png] ![alt text][/home/anooj/mdfileimages/3.png] 
![alt text][/home/anooj/mdfileimages/4.png] ![alt text][/home/anooj/mdfileimages/5.png]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
The code for making predictions on my final model is located in the last cell of the iPython notebook.

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.
![alt text][/home/anooj/mdfileimages/1.png]
Except the 4th image where the model was not able to predict the sign accurately, the rest 4 signs where predicted correctly with 100% accuracy.





