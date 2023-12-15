### Cylinder description

#### Run 1
200 cylinders, each with 100 points uniformly distributed along the surface (without up and bottom plane). Parameters - height and width are random floating in [0, 10). Without noice.

Number of neighbors = 4, C = 1e5.

Takes 1h 16 min to execute. Median of relative error is  0.36.

Graphs are same as in the article.
Left shows embedded Y's got by semidefinite programming solution.
Right shows Y's generated from x by applying Y = xB, where B was obtained by least square method using Y's from the previous step.

![300_cyl_100_points_1_to_10.png](./graphs/200_cyl_100_points_1_to_10.png)



##### Predictive optimization

Median error (norm of difference between original parameters and predicted) is 3.94 on 1000 runs.

To check how good the prediction is I tried computing error for randomly generated parameters amd got median 5.15. So our solution is a little bit better.

#### Run 2
200 cylinders, each with 200 points uniformly distributed. Parameters - height and width are random floating in [0, 10). Noise - normal with standard deviation 0.1.

Number of neighbors = 4, C = 1e5 (C doesn't affect).

Takes 1h to execute. Median of relative error is  0.4.

![200_cyl_200_points_1_to_10.png](./graphs/200_cyl_200_points_1_to_10.png)

##### Predictive optimization

Median error is 4 on 1000 runs.

#### Run 3
200 cylinders, each with 200 points uniformly distributed along the surface (without up and bottom plane). Parameters - height and width are random floating in [0, 10). Noise - normal with standard deviation 0.1.

Number of neighbors = 3, C = 1e5.

Takes 1h 6m to execute. Median of relative error is  0.4.

![200_cyl_200_points_1_to_10_k_3.png](./graphs/200_cyl_200_points_1_to_10_k_3.png)

##### Predictive optimization

Median error is 3.53 on 1000 runs.

#### Run 4
200 cylinders, each with 200 points uniformly distributed along the surface (without up and bottom plane). Parameters - height and width are linspaces of size 200 with arguments `[1, 10]`. Noise - normal with standard deviation 0.05.

Number of neighbors = 3, C = 1e5.

Takes 1h 2m to execute. Median of relative error is  0.4.

![200_cyl_200_points_1_to_10_k_3_determined.png](./graphs/200_cyl_200_points_1_to_10_k_3_determined.png)

##### Predictive optimization

Median error is 3.97 on 1000 runs.

#### Run 5
200 cylinders, each with 200 points uniformly distributed along the surface (without up and bottom plane). Noise - normal with standard deviation 0.05.

Number of neighbors = 2, C = 1e7.

Takes 40 s to execute. Median of relative error is  0.37.

![200_cyl_200_points_1_to_10_k_2_determined.png](./graphs/200_cyl_200_points_1_to_10_k_2_determined.png)

##### Predictive optimization

Median error is 3.12 on 1000 runs.

#### Run 6
200 cylinders, each with 200 points uniformly distributed along the surface (without up and bottom plane).  Noise - normal with standard deviation 0.05.

Number of neighbors = 2, C = 1e5.

Takes 1h 1m to execute. Median of relative error is  0.32.

![200_cyl_200_points_1_to_10_k_2_c_1e5_determined.png](./graphs/200_cyl_200_points_1_to_10_k_2_c_1e5_determined.png)

##### Predictive optimization

Median error is 3.07 on 1000 runs.

#### Run 7
200 cylinders, each with 200 points uniformly distributed along the surface (without up and bottom plane). Noise - normal with standard deviation 0.05.

Number of neighbors = 2, C = 1e4.

Takes 25m to execute. Median of relative error is  0.36.

![200_cyl_200_points_1_to_10_k_2_c_1e4_determined.png](./graphs/200_cyl_200_points_1_to_10_k_2_c_1e4_determined.png)

##### Predictive optimization

Median error is 3.84 on 1000 runs.


#### Run 8
200 cylinders, each with 200 points uniformly distributed along the surface (without up and bottom plane). Noise - normal with standard deviation 0.05.

Number of neighbors = 2, C = 1e3.

Takes 54m to execute. Median of relative error is  0.18.

![200_cyl_200_points_1_to_10_k_2_c_1e3_determined.png](./graphs/200_cyl_200_points_1_to_10_k_2_c_1e3_determined.png)

##### Predictive optimization

Median error is 2.59 on 1000 runs.

#### Run 9
200 cylinders, each with 200 points uniformly distributed along the surface (without up and bottom plane).  Noise - normal with standard deviation 0.05.

Number of neighbors = 2, C = 500.

Takes 8m to execute. Median of relative error is  0.07.

![200_cyl_200_points_1_to_10_k_2_c_500_determined.png](./graphs/200_cyl_200_points_1_to_10_k_2_c_500_determined.png)

##### Predictive optimization

Median error is 2.3 on 1000 runs.

#### Run 10
200 cylinders, each with 200 points uniformly distributed along the surface (without up and bottom plane).  Noise - normal with standard deviation 0.05.

Number of neighbors = 3, C = 500.

Takes 32m to execute. Median of relative error is  0.15.

![200_cyl_200_points_1_to_10_k_3_c_500_determined.png](./graphs/200_cyl_200_points_1_to_10_k_3_c_500_determined.png)

##### Predictive optimization

Median error is 2.93 on 1000 runs.

#### Run 11
200 cylinders, each with 200 points uniformly distributed along the surface (without up and bottom plane).  Noise - normal with standard deviation 0.05.

Number of neighbors = 2, C = 300.

Takes 40s to execute. Median of relative error is  0.07.

![200_cyl_200_points_1_to_10_k_2_c_300_determined.png](./graphs/200_cyl_200_points_1_to_10_k_2_c_300_determined.png)

##### Predictive optimization

Median error is 2.39 on 1000 runs.

#### Run 12
200 cylinders, each with 200 points uniformly distributed along the surface (without up and bottom plane).  Noise - normal with standard deviation 0.05.

Number of neighbors = 2, C = 400.

Takes 1m 40s to execute. Median of relative error is  0.06.

![200_cyl_200_points_1_to_10_k_2_c_400_determined.png](./graphs/200_cyl_200_points_1_to_10_k_2_c_400_determined.png)

##### Predictive optimization

Median error is 2.59 on 1000 runs.

#### Run 12
200 cylinders, each with 200 points uniformly distributed along the surface (without up and bottom plane).Noise - normal with standard deviation 0.05.

Number of neighbors = 2, C = 350.

Takes 1m 40s to execute. Median of relative error is  0.056.

![200_cyl_200_points_1_to_10_k_2_c_350_determined.png](./graphs/200_cyl_200_points_1_to_10_k_2_c_350_determined.png)

##### Predictive optimization

Median error is 2.82 on 1000 runs.



#### Run 13
200 cylinders, each with 200 points uniformly distributed along the surface (without up and bottom plane). No noise.

Number of neighbors = 2, C = 380.

Takes 1m 40s to execute. Median of relative error is  0.068.

![200_cyl_200_points_1_to_10_k_2_c_380_determined_no_noise.png](./graphs/200_cyl_200_points_1_to_10_k_2_c_380_determined_no_noise.png)

##### Predictive optimization

With noise in test data 0.05: median error is 2.43 on 1000 runs.
Without noise: 2.47, som almost same

#### Run 14
500 cylinders, each with 100 points uniformly distributed along the surface (without up and bottom plane).  Noise 0.5.

Number of neighbors = 5, C = 400.

Takes 13 h (!) to execute. Median of relative error is  0.39.

![500_cyl_100_points_1_to_10_k_5_c_400_determined.png](./graphs/500_cyl_100_points_1_to_10_k_5_c_400_determined.png)

##### Predictive optimization

Median error is 2.73 on 1000 runs.


#### Run 15
300 cylinders, each with 100 points uniformly distributed along the surface (without up and bottom plane).Noise 0.5.

Number of neighbors = 5, C = 400.

Takes 3 h to execute. Median of relative error is  0.29.

![300_cyl_100_points_1_to_10_k_5_c_400_determined.png](./graphs/300_cyl_100_points_1_to_10_k_5_c_400_determined.png)

##### Predictive optimization

Median error is 3.54 on 1000 runs.

#### Run 16
400 cylinders, each with 200 points uniformly distributed along the surface (without up and bottom plane).  Noise 0.5.

Number of neighbors = 5, C = 400.

Takes 26m to execute. Median of relative error is  0.13.

![400_cyl_200_points_1_to_10_k_5.png](./graphs/400_cyl_200_points_1_to_10_k_5.png)

##### Predictive optimization

Median error is 3.11 on 1000 runs.

#### Run 17
400 cylinders, each with 300 points uniformly distributed along the surface (without up and bottom plane). Noise 0.5.

Number of neighbors = 5, C = 400.

Takes 35m to execute. Median of relative error is  0.079.

![400_cyl_300_points_1_to_10_k_5.png](./graphs/400_cyl_300_points_1_to_10_k_5.png)

##### Predictive optimization

Median error is 2.98 on 1000 runs.

#### Run 18
400 cylinders, each with 400 points uniformly distributed along the surface (without up and bottom plane).  Noise 0.5.

Number of neighbors = 5, C = 400.

Takes 33m to execute. Median of relative error is  0.078.

![400_cyl_400_points_1_to_10_k_5.png](./graphs/400_cyl_400_points_1_to_10_k_5.png)

##### Predictive optimization

Median error is 2.71 on 1000 runs.


#### Run 19
600 cylinders, each with 600 points uniformly distributed along the surface (without up and bottom plane).  Noise 0.5.

Number of neighbors = 5, C = 400.

Takes 1h 39 to execute. Median of relative error is  0.13.

![600_cyl_600_points_1_to_10_k_5.png](./graphs/600_cyl_600_points_1_to_10_k_5.png)

##### Predictive optimization

Median error is 2.56 on 1000 runs.


#### Run 20
700 cylinders, each with 700 points uniformly distributed along the surface (without up and bottom plane).  Noise 0.5.

Number of neighbors = 6, C = 400.

Takes 3h 6m to execute. Median of relative error is  0.08.

![700_cyl_700_points_1_to_10_k_6.png](./graphs/700_cyl_700_points_1_to_10_k_6.png)

##### Predictive optimization

Median error is 2.52 on 1000 runs.