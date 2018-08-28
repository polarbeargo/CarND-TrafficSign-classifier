# Traffic Sign Recognition #    

## Writeup Template ##

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./traffic-signs-data/test/image1.jpg "Traffic Sign 1"
[image2]: ./traffic-signs-data/test/image2.jpg "Traffic Sign 2"
[image3]: ./traffic-signs-data/test/image3.jpg "Traffic Sign 3"
[image4]: ./traffic-signs-data/test/image4.jpg "Traffic Sign 4"
[image5]: ./traffic-signs-data/test/image5.jpg "Traffic Sign 5"
[image6]: ./traffic-signs-data/test/image6.jpg "Traffic Sign 6"
[image7]: ./traffic-signs-data/test/image7.jpg "Traffic Sign 7"
[image8]: ./traffic-signs-data/test/image8.png "Traffic Sign 7"
### Rubric Points ###   

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README ###
    

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Polarbeargo/CarND-TrafficSign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)
    
    
### Data Set Summary & Exploration ###    

1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pickle library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is ( 34799, 32, 32, 3)
* The number of unique classes/labels in the data set is 43
    
    
2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set using Matplotlib and pandas plot. It is a bar chart plotting traffic sign images and the count of each sign.

![][image8]

### Design and Test a Model Architecture ### 

1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

* As a first step to pre-process data, I use sklearn shuffle() API to shuffle arrays in a consistent way to do random permutations of the collections. The second step, I use train_test_split() API to split arrays into random train and validation subsets and last step I normalized the image data.
    
    
2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Using the transfer learning skill base on pre-trained Neural Network LeNet function learn how to use the dropout() API.

Modified LeNet function consisted of the following layers:

* Convolution layer 1. The output shape should be 28x28x6.
* Activation 1. RELU.
* Pooling layer 1. The output shape should be 14x14x6, Kernel size = 2x2
              Strides = 2x2
              Padding = VALID.
              Dropout with 0.5
* Convolution layer 2. The output shape should be 10x10x16.
* Activation 2. RELU.
* Pooling layer 2. The output shape should be 5x5x16, Kernel size = 2x2
              Strides = 2x2
              Padding = VALID.
              Dropout with 0.5
* Flatten layer. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D, this should have 400 outputs.
* Fully connected layer 1. This should have 120 outputs.
* Activation 3. RELU, Dropout with 0.5.
* Fully connected layer 2. This should have 84 outputs.
* Activation 4. RELU, Dropout with 0.5.
* Fully connected layer 3. This should have 43 outputs.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| input 32x32x3 RGB image   					| 
| Convolution      	    | output shape should be 28x28x6	            |
| RELU					|												|
| Max pooling	      	| stride of 2				                    | 
| Convolution 	        | output shape should be 5x5x16      			|
| RELU					|												|
| Max pooling	      	| stride of 2                                   |
| Flatten               | 400 units                                     |
| Fully connected		| 120 units        							    |
| Fully connected		| 84 units        							    |
| Fully connected		| output 43 units        						|
    
3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* The LeNet was trained with the Adam optimizer for managing learning rate, batch size = 128 images, initial learning rate was 0.001  The model was trained for 10 epochs with one dataset.
Variables were initialized with using of a truncated normal distribution with mu = 0.0, sigma = 0.1 mean=0 and std dev=0.1. Biases initialized with zeros. Learning rate was finetuned by try and error process.
Traffic sign classes were coded into one-hot encodings.
To train the model, I run the training data through the training pipeline to train the model.
Before each epoch, shuffle the training set.
After each epoch, measure the loss and accuracy of the validation set.
Save the model after training.    

4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

* Trained with LeNet Architecture as the problem is similar to image classification problem on which LeNet was applied. The model had been tweaked to handle RGB channel and 43 output classes. 

My final model results were:
* training set accuracy of 98%
* validation set accuracy of 96%
* test set accuracy of 87%
* Use train_test_split() API to split arrays into random train and validation subsets
* Dropouts was used to prevent overfitting in Deep Neural Networks
* Evaluated model by using the validation set. Model predictions and validation classes were compared to get the accuracy of the model on the validation set. Validation set accuracy and training set accuracy are printed at each epoch.
 
### Test a Model on New Images ###    

1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7]

The last image might be difficult to classify because It's Taiwan's traffic sign not belongs to the German traffic signs data set.   

2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry              | No entry                                      | 
| Priority road    		| Priority road									|
| Yield					| Yield											|
| Keep right            | Keep right                                    |
| General caution	    | General caution					 			|
| Stop		            | Stop      							        |
| Taiwan pedestrian     | No passing for vehicles over 3.5 metric tons  |


 Data above shows that the model correctly predicted 6 out of 7 images giving 86% accuracy (compare to 87.6% testing set accuracy of the model). It correctly predicted the images that were already in the class list except image 7 I took from taiwan traffic sign. Correctly predicted images are:

 * image1.jpg - No entry
 * image2.jpg - Priority road
 * image3.jpg - Yield
 * image4.jpg - Keep right
 * image5.jpg - General caution
 * image6.jpg - Stop





