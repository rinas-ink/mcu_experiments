## Maximum Covariance Unfolding Regression

### Goal

Extend a [viny SLAM](https://sci-hub.se/10.1109/IROS.2017.8206595) algorithm in 3d space that runs on low-cost devices as robust, accurate and fast as state of the art algorithms, but requiring fewer resources to run.

### Approaches

[Maximum Covariance Unfolding Regression Article](https://arxiv.org/pdf/2303.17852.pdf)

[Unsupervised Learning of Image Manifolds by Semidefinite Programming](https://sci-hub.se/10.1007/s11263-005-4939-z)

Hypothesis: This approaches reduces the dimensionality of features in the map if the map is represented by a point cloud or a picture

### How can we use hypotheses

 * Usually a lot of resources are spent on loop detection, scan matching is used for that. We can try to use mcu to detect a place, where we have already been + understand the angle of view
 * We can do an initial "check" with less accurate, but faster methods, e.g. computer vision methods, match part of the image and part of the point cloud and analyse part of the point cloud in more detail
   
 -- The image can be compressed or segmented to reduce processing time.
 
 -- Find out what exactly is the angle using a camera and its exact characteristics using lidar 
 
When it can be useful:
 
 -- there is a lot of free space in front of the robot, there are corners on the right and left side
 
 -- too many corners, robots needs to quickly understand where to move. The robot divides the picture into parts of the possible direction of movement, analyses and takes the first suitable one
 
 * try to use mcu based on the picture or try the Weinberger and Saul algorithm that works with the picture.
