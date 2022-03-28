# Probabilistic AI course projects

Projects for successful completion of the course *Probabilistic Artifical Intelligence* led by Prof. Andreas Krause at ETH Zurich, winter semester 2021. Partial code skeletons were provided and core logic and algorithms were implemented by the students. Evaluation was performed on black-box test data in a fixed environment, with a pass for successfully beating the performance baselines.

## Project 0
Very simple exact bayesian inference outputting posterior probabilities, given fixed prior probabilities for sampling from three known distributions (Normal, Laplace, Student).  

*Baseline evaluation:* Hellinger distance for posterior inference on 50 random datasets sampled from the DGP. 

## Project 1
Gaussian Process regression to model air pollution and predict fine particle concentration at new coordinate locations.  
*Challenges:* model selection, GP inference complexity O(n<sup>3</sup>) and asymmetric cost handling. 

Two different implementations that passed the baseline are proposed with `scikit-learn` and `gpytorch` respectively.

**Method #1** (`scikit-learn`): GP regressor with optimal kernel selection via custom k-fold CV, K-means clustering to reduce sample size, custom prediction adjustment to adapt to asymmetric costs.  
**Method #2** (`gpytorch`): GP regressor with structured kernel interpolation, custom prediction adjustment to adapt to asymmetric costs.

*Baseline evaluation:* asymmetric MSE cost on the test set.

![Text](figures/fig1_project1.png)  
*Fig. 1.1 (L to R):* 2D GP predictions map, 3D raw GP estimates (height = GP means, color = GP std devs), 2D GP std dev map.

## Project 2
Bayesian neural network implementation based on the ’Bayes by Backprop’ algorithm by Blundell et al. (2015) [[1]](https://proceedings.mlr.press/v37/blundell15.html) for multi-class classification on the MNIST dataset. The test set contains added uncertainty via pixelation and random rotations. Additional calibration measurements of predictions (see Guo et al. (2017) [[2]](http://proceedings.mlr.press/v70/guo17a.html)).  
*Challenges:* algorithm implementation, weight prior and variational posterior selection, calibration considerations.

*Baseline evaluation:* compound score of both accuracy and empirical expected calibration error (ECE).

mnist_least/most confident  
*Fig. 2.1:* Sample test set most and least confident predictions for MNIST.  
fashionmnist_least/most confident  
*Fig. 2.2:* Sample test set most and least confident predictions for FashionMNIST as a sample visualisation of the BNN’s calibration under strong distribution shift. 

## Project 3
Implementation of Bayesian optimization under constraints to the feasible domain (2D grid), following Gelbart et al. (2014) [[3]](https://arxiv.org/abs/1403.5607).  
*Challenges:* joint training of constraint and objective, acquisition function choice, constraint satisfaction. 

*Baseline evaluation:* mean normalized regret under constraint satisfaction on 27 different tasks. 

![Text](https://github.com/alextimans/eth-probabilistic-ai/blob/main/figures/project3_fig1.png)  
*Fig. 3.1:* Some solution features for the toy example. *(L to R)* Objective function map with feasability domains, 3D plot of objective function estimate, 3D plot of constraint function estimate, constraint function map with feasability domains.

## Project 4
Reinforcement learning task using Generalized Advantage Estimation (GAE) as presented in Schulman et al. (2016) [[4]](https://arxiv.org/abs/1506.02438). It is a model-free policy gradient approach with two neural networks as actor and critic respectively. The control task is to learn a policy to smoothly descend a lunar lander to the ground in between two flags with minimal fuel use and without damaging it.  
*Challenges:* dual neural network parametrizations, improving rewards structure, extending vanilla policy gradients with advantage estimation.

*Baseline evaluation:* estimated expected cumulative reward of the final policy over an episode. 

GIF  
*Fig. 4.1:* Sample visualisation of the final policy on the lunar lander control task at evaluation time.
