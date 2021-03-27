# General Info

This is an implementation of the hybrid model described in the scientific paper "Two novel hybrid Self-Organizing Map based emotional learning
algorithms"[[1]](#1). 

Part of the DSML Master's thesis that can be found [here](http://artemis.cslab.ece.ntua.gr:8080/jspui/).

# Technologies
* Python 3.8.3
* PyTorch 1.7.1
* Numpy 1.20.1
* QuickSOM 0.0.4

# Files
* customSOM.py: Extends SOM class of quicksom library.
* EmGD.py: Extends the SGD class of Pytorch and implements the emotional backpropagation algorithm.
* EmSOM.py: Structures a custom neural network module by extending the nn.module of PyTorch.
* main.py: Script for training and testing of the hybrid network.


## References
<a id="1">[1]</a> 
Qun Dai, Lin Guo (2019). 
Two novel hybrid Self-Organizing Map based emotional learning
algorithms. 
Neural Computing & Applications, 31, 2921–2938.

