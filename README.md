# Sensor Fusion Engineer Project 3 - 3D Object Tracking

## Benjamin Söllner

This project is forked from the [Udacity Sensor Fusion Nanodegree](https://www.udacity.com/course/sensor-fusion-engineer-nanodegree--nd313) online class content and subsequently completed to meet the courses project submission standards. The remaining section of this `README` includes the Reflection which has to be as part of this project and details about the general course content and how to build this project. The source code in this repo also contains the lesson quizzes in the separate ``src/quizzes`` folder. Go to [udacity/SFND_3D_Object_Tracking](https://github.com/udacity/SFND_3D_Object_Tracking) if you want to retrieve the original (unfinished) repo. Don't you cheat by copying my repo in order to use it as your Nanodegree submission! :-o

## Reflection

This section answers how this Udacity project submission fulfils the project [rubric](https://review.udacity.com/#!/rubrics/2550/view).

### FP.1 Match 3D Objects

Bounding Boxes are matched between images in ``matchBoundingBoxes`` by utilizing the ``cv::DMatch``es: first, a ``map`` is used to count how many points of the bounding boxes in the previous data frame are present in the current one. This ``map`` is then transformed into a vector of tuples containing ...

* the bounding box ID of the previous data frame
* the bounding box ID of the current data frame
* the number of matching points

... which is then sorted by a comperator function comparing the last element of the tuple (number of matching points).

The sorted vector can easily be traversed and an auxiliary ``set`` can ascertain that each bounding box of the previous frame is only mapped exactly once (due to the sorting with the highest number of matches).

### FP.2 Compute Lidar-based TTC

``computeTTCLidar`` processes the lidar points of each bounding box. In order to remove outliers, small clusters of lidar points are removed using ``pcl::EuclideanClusterExtraction`` implemented in the auxiliary ``euclideanClustering(...)`` function as seen in the Lidar Sensor Fusion course.

### FP.3 Associate Keypoint Correspondences with Bounding Boxes

``clusterKptMatchesWithROI`` traverses all ``cv::DMatch``es and checks for every ``trainIdx``, which are the keypoint indices in the current frame, whether the ``boundingBox`` contains that point and - if so - adds it to the ``boundingBox.kptMatches``.

### FP.4 Compute Camera-based TTC

In ``computeTTCCamera``, first, two nested loops iterate through all possible pairs of ``kptMatches``. For each pair, the the the distance ratio in the current and previous frame are computed - subsequently, after sanity checks (minimum distance and division by zero) the distance ratio is computed (``distanceCurr``/``distancePrev``).

All ``distanceRatios``s are stored in a vector which, in a second step, is sorted. The median now can easily be retrieved by retrieving the (two) element(s) in the middle. Now, ``TTC`` can be calculated.

### FP.5 & FP.6 Performance Evaluation 1 & 2

**NOTE**: all charts below can be explored interactively on [Tableau Public](https://public.tableau.com/profile/benjamin.s.llner#!/vizhome/chart_15888399888880/Project33DObjectTrackingofUdacitysSensorFusionEngineerNanodegree).

From the images, it is evident that the vehicle comes closer from about frame 10ff. However, both TTC from Lidar as well as TTC from Camera get noisier. This is due to the increasing number of lidar points. Even small fluctuations in lidar points when the speed is slow can lead to large noise levels, especially if the car is close. From about frame 48, lidar data seems to become quite useless, intermittently no lidar points at all can be retrieved, which leads to faulty TTC computations.

![TTC Lidar and TTC Camera as well as Number of Lidar Points over time](stats/lidar_vs_camera.png)

All combinations of detector/descriptor/matcher algorithms have been tried. There are many faulty TTC estimations due to the low number of descriptors. After frame 48, the consistency of keypoint matches becomes unstable.

![TTC for Lidar and all camera algorithms as well as Number of Lidar Points and Key Point Matches over time](stats/all_algorithms.png)

There are some robust combinations as well, the below image displays some of them. From the midterm project, we can order them per runtime in ascending order:
* (FAST, BRIEF, MAT_BF) - **fastest**
* (FAST, FREAK, MAT_BF) - **most accurate**
* (SIFT, FREAK, MAT_BF)
* (FAST, BRISK, MAT_BF)

![TTC for Lidar and stable camera algorithms and Key Point Matches over time](stats/stable_algorithms.png)

## Course Content

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.
