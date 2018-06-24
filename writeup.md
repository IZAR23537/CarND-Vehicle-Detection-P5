**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Vehicles.JPG
[image2]: ./output_images/Non-vehicles.JPG
[image3]: ./output_images/HOG_image.JPG
[image4]: ./output_images/boxes.JPG
[image5]: ./output_images/Heatmap.JPG
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explaining how I extracted HOG features from the training images.
 

Im the second code cell of the IPython notebook I started reading all the `vehicle` and `non-vehicle` images. 
Here are some examples of the `vehicle` and `non-vehicle` images:

![Vehicles][image1]
![Non-vehicles][image2]

I then explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  
I grabbed a random image of the vehicle class and displayed it to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`:


![HOG_image][image3]



#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters (and lots of trying) in order to find cars on an image (and not getting false-positive results). 
In the end I used the following parameters: 
* orientations=12;
* pixels_per_cell=(16, 16);
* cells_per_block=(2, 2).


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in the 14th code cell of the IPython notebook. 
First, I extracted car and non-car features using the following parameters: 
	YCrCb colorspace; 
	HOG features: orientation = 12, pixels_per_cell = (16, 16), cells_per_block = (2, 2), hog_channel = 'ALL';
	Spatial size = (32, 32)
	Hist_bins = 32

Second, I created an array stack of feature vectors and defined labels vector. Splitted the data into randomized training and test sets.
I tested the SVC accuracy with the splitted test set. The accuracy was around 98%.


### Sliding Window Search

#### 1. Describe how I implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the 17th code cell of the IPython notebook I used the find_car function which was provided by Udacity. 
I modified this function by changing the image search area to search for cars in the right side of the image. Also this function returns a box area list where it found a car.



![Boxes][image4]

#### 2. Showing some examples of test images to demonstrate how the pipeline is working.  What did I do to optimize the performance of my classifier?

I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.
After marking the found features on the given test image, I used heat-map and heat map treshold in order to sort out the false positive results.

Here are an example image:

![Heatmap][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I searched right side of the image for cars in order to limit the search area where in the video we would expect cars. (This code can be found in the IPython notebook find_cars function)
I recorded the positions of positive detections in each frame of the video.  
From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  
I then assumed each blob corresponded to a vehicle.  
I constructed bounding boxes to cover the area of each blob detected.
I added the bounding boxes to a history list which I used for creating the next bounding boxes.



### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Overall, it was very difficult to sort out false positive results and finding the right parameters that would work on the provided video.
At the moment, the pipeline is not so great for detecting cars on the left side, because of the limited search are that i provided. 
If I unlimit this searching area it most likely get some false positive results.

I think to make it more efficient I should also try out some deep learning technic in the future.


