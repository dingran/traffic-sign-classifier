# Traffic Sign Classification with LeNet and Multi-Scale CNN

In this project, I experimented with several convolutional neural network (CNN) architectures for traffic sign image classification,
and achieved an accuracy of **99.02%** with a variant of LeNet5 as well as a variant of multi-scale feature CNN discussed in 
[Sermanet / LeCunn paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). The CNN is implemented with **Tensorflow**, 
data preprocessing and augmentation are done with tools in **OpenCV**.

This project is part of Udacity Self-Driving Car Nanodegree (Term1 in May 2017). The project instruction is in [README_udacity.md](README_udacity.md) 
and the dataset can be downloaded here at [traffic-signs-data.zip](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)

---


## Table of Contents

to be inserted


---

# Introduction

## Goals

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Model Results Preview

Before we dive into the details, here is a quick preview of the models' performance on the (augmented) validation set and the original test set.

| Model | Parameter set | No. of trainable parameters | Validation accuracy | Test accuracy | 
| :---: | ---: | ---: | ---: | ---: | 
| lenet | standard | 139,391  | 95.48% | 97.14% | 
| lenet | big | 2,137,739  | 98.36% | **99.02%** |
| sermanet | standard | 1,681,359| 98.27% | 98.82% |
| sermanet_v2 | standard | 3,972,139  | 98.18% | 98.35% |
| sermanet | big | 4,351,643  | 98.60% | **99.02%** |

The best results are obtained with model "lenet" (a variant of LeNet5) 
and "sermanet" (a variant of the multi-scale CNN proposed by [Sermanet / LeCunn paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)) with 
corresponding parameter sets that result in a large networks.

In testing with additional images screen captured from Google Street View, both networks

These two models will be referred to as **lenet** and **sermanet** in this report. 
Both models will be reported in corresponding HTML reports listed below and both trained netowrks returned

## Files
Here is a summary of the key folders and files in this project

* README:
    * The file you are reading is the project writeup
    * The original README from udacity is [README_udacity.md](README_udacity.md), which contains project instructions
* Code:
    * [Traffic_Sign_Classifier.ipynb](Traffic_Sign_Classifier.ipynb) is Jupyter notebook holding the top-level execution code used in this project
    * [tsc_utils.py](tsc_utils.py) holds all the code for data preprocessing, augmentation, network implementation and model training routines
* Reports:
    * [](Traffic_Sign_Classifier_0613_big_lenet.html)
* [test_images_output](test_videos_output) and [test_videos_output](test_videos_output) hold the labeled images and videos produced by the [P1.ipynb](P1.ipynb)
* [test_images_output_DEBUG](test_videos_output_DEBUG) and [test_videos_output_DEBUG](test_videos_output_DEBUG) 
hold the labeled images and videos produced by the [P1.ipynb](P1.ipynb) in debug mode
* [create_gif.py](create_gif.py) creates the gif used in this report

<!--- comments
lenet orig_lenet 139391
lenet big_lenet 2137739
sermanet standard 1681359
sermanet big 4351643
sermanet_v2 standard 3972139
sermanet_v2 big 15796267
--->




[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


In this project, I prepared a few augmented datasets (i.e. dataset1, dataset2 and dataset3) as listed below. 
All the models listed in the next table are trained with all 3 datasets in sequence.

| Dataset | No. of training examples | No. validation examples | Dataset generation method |
| ------------- | ------------- | ------------- | ------------- |
| dataset0 | 34,799 | 4,410 | Original dataset |
| dataset1 | 59,788 | 7,590  | Based on dataset0, generated additional images by flipping images that have mirror symmetry|
| dataset2 | 1,016,396 | 129,030 | Based on dataset1, add additional images by |
| dataset3 | 1,016,396 | 129,030  | Content Cell  |



### Key files in this repo:



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

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


