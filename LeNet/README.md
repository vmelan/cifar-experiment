# LeNet 

## Description 
The first successful Convolutional Neural Network (CNN) was introduced by LeCun et al. in their 1998 paper, 
*Gradient-Based Learning Applied to Document Recognition* ([link](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)). 

The **LeNet-5** is the latest offering from Yann LeCun and his colleagues and was used for classifying
handwritten digits successfully for Optical Character Recognition (OCR) based activities such as reading
ZIP codes, checks, and so on.

## Architecture 
The architecture of the CNN is presented in Figure 2 of the paper: 
<p align="center">
	<img src="images/lenet.PNG" width=75% height=75%>
	<br>Figure : Architecture of LeNet-5
</p>
While the output units are Radial Basis Function (RBF) units, in the implementations, we will 
prefer to use the Softmax function. 

A more in-depth overview of the layers composed in this architecture is show in the table below:
<br>

| Layer Type |  Output Size | Filter size / Stride | Activation |
|:----------:|:------------:|:--------------------:|:----------:|
|    Input   |  28 x 28 x 1 |           -          |      -     |
|    Conv    |  28 x 28 x 6 |       5 x 5 / 1      |    tanh    |
|  Avg Pool  |  14 x 14 x 6 |       2 x 2 / 2      |      -     |
|    Conv    | 10 x 10 x 16 |       5 x 5 / 1      |    tanh    |
|  Avg Pool  |  5 x 5 x 16  |       2 x 2 / 1      |      -     |
|   Flatten  |     400      |           -          |      -     |
|     FC     |      120     |           -          |    tanh    |
|     FC     |      84      |           -          |    tanh    |
|   Output   |      10      |           -          |   softmax  |