## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images/undistorted_board.png "Undistorted"
[image2]: ./images/Undistorted.png "Road Undistorted"
[image3]: ./images/warped.png "Warp Example"
[image4]: ./images/binarize.png "Binarize Example"
[image5]: ./images/shadow_detect.png "shadow"
[image6]: ./images/detect_lane_lines.png "detect lanes"
[image7]: ./images/unwarped_lane_lines.png "unwarped"
[image8]: ./images/shadow_detect_wm.png "undetectable"
[video1]: ./result_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Camera Calibration

#### Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

The code for this step is contained in the first code cell of the IPython notebook located in "./AdvancedLaneLines.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Apply a distortion correction to raw images.

Firstly, I apply the distortion correction using cv2.undistort, camera calibration matrixes(first cell of IPython notebook).
![alt text][image2]

#### 2. Apply a perspective transform to rectify raw image ("birds-eye view").

In my solution, to reduce the noise I apply perspective transform before the thresholding binarize.
The code for my perspective transform includes a function called `get_perspective_transformed_image()`, which appears in the 3rd code cell of the IPython notebook.
The `get_perspective_transformed_image()` function takes as inputs an `image`, as well as `source` and `destination` points.
I chose the hardcode the source and destination points and setting margins in the following manner:

```python
imshape = image.shape # 720, 1280, 3
lr_margin = 300 # left and right margin
tb_margin = 30 # top and bottom margin
dst_margin = 350 # destination

source = np.float32([
    [(lr_margin, imshape[0]-tb_margin),
    (600, 450),
    (680, 450),
    (imshape[1]-lr_margin, imshape[0]-tb_margin)]])

destination = np.float32([
    [dst_margin, imshape[0]],
    [dst_margin, 0],
    [imshape[1] - dst_margin, 0],
    [imshape[1] - dst_margin, imshape[0]]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 300, 690      | 350, 720      |
| 600, 450      | 350, 0        |
| 680, 450      | 930, 0        |
| 980, 690      | 930, 720      |

I verified that my perspective transform was working as expected by drawing the `source` and `destination` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

#### 3. Use color transforms, thresholding absolute value of sobel and magnitude of sobel to create a thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image (9, 10th code cell of the IPython notebook).
Here's an example of my output for this step.

![alt text][image4]

#### 4. Shadow detection and removal.

According to this paper[1], the grayscale value of shadowed area is larger than ratio image value(`(h channel + 1)/ (v channel + 1)`), but on the contrary grayscale value of non-shadowed area is lower than ratio image value.
So, I applied hsv thresholding and Otsu's thresholding to detect and remove shadol in image (12, 13th cell of the IPython notebook).

The following figure is example of shadow detection and removal.

![alt text][image5]

#### 5. Use sliding window search to identified lane line pixels and fit their positions with a polynomial

Firstly, I create histgram from preprocessed image, and find left and right starting point of window search at the bottom of the image. Then, I setting window on these points and begin to implement sliding window search. In this process, I collect nonzero pixel positions within window as candidate for lane lines. If I can detect over 50 nonzero pixels within window, the window slide toward top. In contrast, if I can't detect over 50 nonzero pixel within window, recenter the next window on mean position of nonzero x within window. When the window reached on top of image, sliding window is finish, and apply the polynomial fit using collected nonzero coordinates.

Here's an example of my output for this step (15th cell of the IPython notebook, `sliding_window_search()` at `AdvLaneLine.py`).
![alt text][image6]

#### 6. Determine the curvature of the lane and vehicle position with respect to center.

To measure the curvature of detected lane lines, I calculated the radius of curvature using mean lane lines.
I also calculated the point of y=720 using mean lane lines to take center of lane, and I used center of image(640) as center of vehicle position. I measured the vehicle position with respect to center the by subtracting these values.
In addition to these process, I assumed that between lane lines was 3.7 meter and converted pixels to meter.

I did this in my code in `AdvLaneLine.py` in the function `determine_curvature(), get_vehicle_position()`.

#### 7. Warp the detected lane boundaries back onto the original image.
I implemented this step in my code in the function `get_perspective_transformed_image()`.
Here is an example of my result on a test image(18th cell of the IPython notebook):

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./result_project_video.mp4)

---

### Discussion

##### For more robunst lane line detection

The troubles I faced in this project is to detect noises from the condition of road.
I could detect and remove the shadow on road, but could not detect noises like a changing color of surface of road.
If I can detect these noises, more robust lane line detection will be implement.

![alt text][image8]

### References
[1] : N.N. Ahmed Salim, X. Cheng and X. Degui, 2014. A Robust Approach for Road Detection with Shadow Detection Removal Technique. Information Technology Journal, 13: 782-788.
