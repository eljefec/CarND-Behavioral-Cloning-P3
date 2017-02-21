#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* load.py for loading training data
* pre.py for preprocessing training data
* build.py for building a neural network
* history.py for dumping Keras history object
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

Code file: build.py

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (build.py lines 63-67) 

The model includes RELU layers to introduce nonlinearity (code lines 45, 73, 75, 77). The data is normalized in the model using a Keras lambda layer (code line 56). 

The model crops out the top 50 pixels and the bottom 20 pixels to ignore the sky and car hood respectively (code line 53).

The model has a Color Space Transformation layer to allow the model to learn the best color representations (code line 56).

####2. Attempts to reduce overfitting in the model

To reduce overfitting, the model contains dropout layers before the final fully connected layers (build.py lines 72, 74, 76, 78). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 58). 

I tested the model by running it through the simulator and ensuring that the vehicle stayed on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 47).

####4. Appropriate training data

I used the training data provided by Udacity. It is mostly center lane driving. I included examples of turning left at the fork between pavement and dirt roads.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple model then improve it through architecture tweaks and training data augmentation.

I originally started with a LeNet architecture to confirm that my training pipeline worked. The trained model performed terribly. It turns out that I was training my model with RGB images whereas the driving script was receiving BGR images from the simulator. I fixed this by using 'cv2.imread'.

To prove that my training pipeline worked with BGR images, I started with a very simple model of mostly fully connected layers. See function 'build_model_simple' in build.py (code lines 21-29). This simple model was successful in driving basically, but it was not good enough to drive the whole track.

I read some Udacity forum posts that other students had success using the Nvidia model described in 'http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf'.

I started with a convolutional neural network similar to the Nvidia model and trained it on the data provided by Udacity, but only the center camera images. The model was able to drive more of the track, but it had difficulty at the fork between pavement and dirt road.

I proceeded to augment the training data. 

I flipped all images horizontally (and negated their corresponding steering angles) to double my training set size. This also balanced the bias in the data towards left turns such that left and right turns were equally represented.

I incorporated the left and right camera images and used them as training examples by modifying the actual steering angle with an appropriate correction value (this ended up being 0.04 in my final model).

To improve behavior at the pavement-dirt fork, I recorded several center-lane driving examples of turning left at the fork. When I included these examples, it was not sufficient to influence my model to prefer the pavement side of the fork. I included my fork dataset 5 times, and found that was successful at correcting my model at the fork.

I now had to tune my hyperparameters: steering correction value, dropout probability, and number of epochs. I experimented with these hyperparameters by training models with varying hyperparameter values then running the models in the simulator. I tried steering correction values between 0.01 and 0.25 and dropout probabilities between 0.1 and 0.5.

During this experimentation process, I modified the training pipeline to incorporate Keras callbacks EarlyStopping and ModelCheckpoint.

My model was able to drive successfully around the track with these hyperparameters:
1. Steering correction: 0.04
2. Dropout probability: 0.2
3: Number of epochs: 13 (with early stopping)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

## TODO below

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
