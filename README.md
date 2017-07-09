# Traffic Sign Classification with LeNet and Multi-Scale CNN

In this project, I experimented with several convolutional neural network (CNN) architectures for traffic sign image classification,
and achieved an accuracy of **99.02%** with a variant of LeNet5 as well as a variant of multi-scale feature CNN discussed in 
[Sermanet / LeCunn paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). The CNN is implemented with **Tensorflow**, 
data preprocessing and augmentation are done with tools in **OpenCV**.

This project is part of Udacity Self-Driving Car Nanodegree (Term1 in May 2017). The project instruction is in [README_udacity.md](README_udacity.md) 
and the data set can be downloaded here at [traffic-signs-data.zip](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)

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


## Models

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

In testing with additional images screen captured from Google Street View, both networks achieved **100% accuracy over 11 images, 
including one "novel" sample** - a sign in a known category but with different background color and additional symbol on the sign.

These two models will be referred to as **lenet** and **sermanet** in this report. 
Both models will be reported in corresponding HTML reports listed below and both trained netowrks returned

## Files
Here is a summary of the key folders and files in this project

README:
* [README.md](README.md): the file you are reading; this is the project writeup
* [README_udacity.md](README_udacity.md): the original README from udacity which contains project instructions

Code:
* [Traffic_Sign_Classifier.ipynb](Traffic_Sign_Classifier.ipynb) is Jupyter notebook holding the top-level execution code used in this project
* [tsc_utils.py](tsc_utils.py) holds all the code for data preprocessing, augmentation, model implementation training

Reports:
* [Traffic_Sign_Classifier_0709_Big_LeNet.html](reports/Traffic_Sign_Classifier_0709_Big_LeNet.html): 
HTML of Traffic_Sign_Classifier.ipynb using lenet
* [Traffic_Sign_Classifier_0709_Big_Sermanet.html](reports/Traffic_Sign_Classifier_0709_Big_Sermanet.html): 
HTML of Traffic_Sign_Classifier.ipynb using sermanet
* [report.html](report.html): required html report for submission, it is a copy of 
[Traffic_Sign_Classifier_0709_Big_LeNet.html](reports/Traffic_Sign_Classifier_0709_Big_LeNet.html)

<!--- comments
lenet orig_lenet 139391
lenet big_lenet 2137739
sermanet standard 1681359
sermanet big 4351643
sermanet_v2 standard 3972139
sermanet_v2 big 15796267
--->

# Data Set Exploration

Summary statistics of the traffic signs data set calculated with Pandas:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

Normalized histograms of training, validation and testing sets show that the relative frequencies of occurence of different categories are consistent among the three data sets. 

![](writing/sample_distribution_v2.png)

Here are example images from a few classes (a full survey is available in [Traffic_Sign_Classifier.ipynb](Traffic_Sign_Classifier.ipynb))
 
![](writing/example_images.png)


# Design and Test a Model Architecture

## Qualitative Findings

I started with a relatively well-known network (LeNet5) with settings to keep the network size small in order to iterate more quickly on a few topics and get some qualitative intuition. 
While none of these qualitative findings are conclusive, but they are useful initial guidelines

For example: 
* Are grayscale images better or should I use all color channels? What is the best way to "normalize" images regarding brightness, contrast and etc? 
    * See [Data Preprocessing](#data-preprocessing)
* How does the size of the data set compare the the size of the network? 
    * Should I use much much large networks or the starting point of LeNet5 is already work very well?
    * Is it better to "equalize" the sample count in each class or should I keep the relative occurrence as-is?
    * Does having much more augmented data help accuracy?
    * See [Data Autmentation](#data-augmentation)


## Data Pre-processing and Augmentation

### Data Pre-processing
As shown in the previous section, the images within each class have very different brightness, sharpness and color saturation. 
In order to present data to the model in a consistent fashion, therefore to allow the model generalize better into test set.

Here are the steps in the preprocessing pipeline:

1. Covert color images to grayscale images
    * Through trial and error, I observed that using grayscale images seems to achieve better performance on test set. 
    This is might be slight counter-intuitive. It might be because although color images provide more information, the image quality vary greatly and might have offset the benefit.

2. Apply histogram equalization to get uniform contrast
    * Instead of a global histogram equalization (HE), which might have over exposure or under exposure on the region contains useful information, 
    I used a localized adaptive version - Contrast Limited Adaptive Histogram Equalization, with the setting ```cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))```

3. Normalize to zero mean and unity variance
    * This is to normalize the numerical values globally.
    
To illustrate this process, let's take a look at the following example. The first row are the original images. Row 2, 3 and 4 corresponds to the output of
 step 1, 2 and 3 respectively. 
We can see that the preprocessing pipeline is indeed capable of normalizing out different brightness, recovering contrast from poor quality color images, and yielding a set
of more uniform quality images for use in the model.

![](writing/preprocessing.png)


### Data set augmentation

A simple way to generate more data is to take advantage of the mirror symmetry in some of the signs. For example, in the sample images shown above,
if flipped horizontally, all class 33 images would become valid samples for class 34, all class 30 images would still be valid samples for class 30. 
This is implemented by ```flip_extend()``` in [tsc_utils.py](tsc_utils.py). (The credit of this flipping method goes to: https://github.com/navoshta)

In addition, a small change to the image should not affect the classification if the classfier is robust and well generalized. Based on this, I implemented
a set of functions to apply small changes to the sign images, these operations are:
* zoom in or out
* rotate left or right
* shift x and/or y
* sharpen or blur
* random distort (implemented with ```cv2.warpPerspective()``` and randomized corners)

Here is a example image with each of these operations applied 

![](writing/augmentation.png)

The overall function driving the augmentation is ```augment_data()``` in [tsc_utils.py](tsc_utils.py). 
It has a scaling parameter ```factor``` that controls the magnitude of all these operations.


I prepared 3 augmented data sets based on the flipping and augmentation methods discussed above. Only training and vaidation sets are augmented.
All the models listed in the next table are trained with all 3 data sets in sequence, with decreasing learning rates.

Here is a summary of the data sets:

| Data set | N_train | N_validation | Augmentation method |
| --- | ---: | ---: | ------------- |
| dataset0 | 34,799 | 4,410 | Original data set |
| dataset1 | 59,788 | 7,590  | Based on dataset0, applied ```flip_extend()``` |
| dataset2 | 1,016,396 | 129,030 | Based on dataset1, applied ```augment_data()``` with ```factor=1.0``` |
| dataset3 | 1,016,396 | 129,030 | Based on dataset1, applied ```augment_data()``` with ```factor=0.7``` |


## Model Architecture

### LeNet

I used LeNet5 as a starting point due to its good performance on other image classification tasks and relative small network size.

The LeNet model implemented here is slightly different than the classical implementation as summarized below.


| Layer         		|     Description (this project)    			| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|


def lenet(x, params, is_training):
    print(params)
    do_batch_norm = False
    if 'batch_norm' in params.keys():
        if params['batch_norm']:
            do_batch_norm = True

    with tf.variable_scope('conv1'):
        conv1 = conv_relu(x, kernel_size=params['conv1_k'], depth=params['conv1_d'], is_training=is_training,
                          BN=do_batch_norm)
    with tf.variable_scope('pool1'):
        pool1 = pool(conv1, size=2)
        pool1 = tf.cond(is_training, lambda: tf.nn.dropout(pool1, keep_prob=params['conv1_p']), lambda: pool1)
    with tf.variable_scope('conv2'):
        conv2 = conv_relu(pool1, kernel_size=params['conv2_k'], depth=params['conv2_d'], is_training=is_training,
                          BN=do_batch_norm)
    with tf.variable_scope('pool2'):
        pool2 = pool(conv2, size=2)
        pool2 = tf.cond(is_training, lambda: tf.nn.dropout(pool2, keep_prob=params['conv2_p']), lambda: pool2)

    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])
    print('lenet pool2 reshaped size: ', pool2.get_shape().as_list())

    with tf.variable_scope('fc3'):
        fc3 = fully_connected_relu(pool2, size=params['fc3_size'], is_training=is_training, BN=do_batch_norm)
        fc3 = tf.cond(is_training, lambda: tf.nn.dropout(fc3, keep_prob=params['fc3_p']), lambda: fc3)

    with tf.variable_scope('fc4'):
        fc4 = fully_connected_relu(fc3, size=params['fc4_size'], is_training=is_training, BN=do_batch_norm)
        fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, keep_prob=params['fc4_p']), lambda: fc4)

    with tf.variable_scope('out'):
        logits = fully_connected(fc4, size=params['num_classes'], is_training=is_training)

    return logits


### Sermanet



### Sermanet_v2



## Model Training




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
 

# Test Models on New Images

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



