# EE-559 Deep Learning - Project 2 - Implementation of a mini deep-learning framework to train a MLP network

The aim of this project was to build a mini deep-learning framework using only PyTorch's basic tensor operations and to develop with it an MLP that would be able to classify points inside or outside a circle of a given radius. The implemented network is composed of three fixed hidden layers of 25 units. Different loss criterions and activation functions are available in the framework.

To see our work and results, just run the `test.py` file (in src folder), and follow the guidelines of the user-interface we implemented.


## Prerequisites
Python 3.7 <br/>
PyTorch

## Dataset 
The data-set is generated as 1000 points randomly distributed in [0,1]x[0,1]. A point is labeled 1 if it is inside a circle of radius 1/sqrt(2π) centered at (0.5,0.5), and 0 outside.

# Organisation of the repository

```
|
|   README.md                                       > README of the project  
|   report.pdf                                      > LaTex report of project 2
|   
+---figures                                         > plots generated by `Test.py` + plots of the report
|
|   png_files                                       > png format plots
|
+---src                                           
|
|   helpers.py                                      > contains functions to plot, generate data and train model
|   Modules.py                                      > class Modules w/ children Linear, Activation functions and Loss criterions
|   Optimizer.py                                    > class Optimizer w/ SGD available 
|   Test.py                                         > runs our best model / runs model with parameters chosen by the user			                              
|
```  
