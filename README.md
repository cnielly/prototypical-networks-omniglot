# Prototypical Networks on the Omniglot Dataset
An implementation of "Prototypical Networks for Few-shot Learning" on a notebook in Pytorch

## I. Prototypical Networks

**Prototypical Networks** were introduced by Snell et al. in 2017 (https://arxiv.org/abs/1703.05175). 
They started from a pre-existing architecture called **Matching Networks** introduced in a previous paper (https://arxiv.org/abs/1606.04080).
They are both part of a broader family of algorithms called **Metric Learning Algorithms**, 
and the success of these networks is based on their capacity to understand the similarity relationship among samples.

"Our approach is based on the idea that there exists an embedding in which points cluster around a single prototype 
representation for each class." claim the authors of the original paper *Prototypical Networks for Few-shot Learning*) 

In other words, there exists a Mathematical representation of the images, called the **embedding space**, 
in which images of the same class gather in clusters. 
The main advantage of working in that space is that two images that look the same will be close to each other, 
and two images that are completely different will be far away from each other. 

<p align="center">
<img src="https://github.com/cnielly/prototypical-networks-omniglot/blob/master/readme_images/prototypes_new.jpg" width="500" alt="Clusters in the embedding space">
</p>

Here the term "close" refers to a distance metric that needs to be defined. The cosine distance or the Euclidean distance are usually taken. 

Unlike typical Deep Learning architecture, Prototypical Networks do not classify the image directly, and instead learn its mapping in the embedding space. 
To do so, the algorithm does several “loops” called **episodes**. Each episode is designed to mimic the Few-shot task. Let’s describe in detail one episode in training mode:

<ins>**Notations:**</ins>

In Few-shot classification, we are given a dataset with few images per class. N<sub>c</sub> classes are randomly picked, and for each class we have two sets of images: the support set (size N<sub>s</sub>) and the query set (size N<sub>q</sub>). 

![Representation of one sample](https://github.com/cnielly/prototypical-networks-omniglot/blob/master/readme_images/sample_representation.JPG)

<ins>**Step 1: embed the images**</ins>

First, we need to transform the images into vectors. This step is called the embedding, and is performed thanks to an "Image2Vector" model, which is a Convolutional Neural Network (CNN) based architecture.

<ins>**Step 2: compute class prototypes**</ins>

This step is similar to K-means clustering (unsupervised learning) where a cluster is represented by its centroid. 
The embeddings of the support set images are averaged to form a class prototype.

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\large&space;v^{(k)}&space;=&space;\frac{1}{N_{s}}&space;\sum_{i=1}^{N_{s}}&space;f_{\Phi&space;}(x_{i}^{(k)})" title="\LARGE v^{(k)} = \frac{1}{N_{s}} \sum_{i=1}^{N_{s}} f_{\Phi }(x_{i}^{(k)})" />
</p>

v<sup>(k)</sup> is the prototype of class k.

<ins>**Step 3: compute distance between queries and prototypes**</ins>

This step consists in classifying the query images. To do so, we compute the distance between the images and the prototypes. Metric choice is crucial, and the inventors of Prototypical Networks must be credited to their choice of distance metric. They noticed that their algorithm and Matching Networks both perform better using Euclidean distance than when using cosine distance. 

Cosine distance             |  Euclidean distance
:-------------------------:|:-------------------------:
![](https://latex.codecogs.com/png.latex?\large&space;d\\_cos(v,&space;q)&space;=&space;\frac{v\cdot&space;q}{\left&space;\\\|&space;v&space;\right&space;\\\|\left&space;\\\|&space;q&space;\right&space;\\\|}&space;=&space;\frac{\sum&space;v_iq_i}{\sqrt{\sum&space;v_i^2}&space;\sqrt{\sum&space;q_i^2}})  |  ![](https://latex.codecogs.com/png.latex?\large&space;d\\_eu(v,q)&space;=&space;\left&space;\\\|&space;v-q&space;\right&space;\\\|&space;=&space;\sqrt{\sum&space;(v_i-q_i)^2})

Once distances are computed, a softmax is performed over distances to the prototypes in the embedding space, to get probabilities. 

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\large&space;p_\Phi(y=k|x)=\frac{exp[-d(f_\Phi(x),v^{(k)})]}{\sum_{k'=1}^{N_c}&space;exp[-d(f_\Phi(x),v^{(k')})]}" title="\large p_\Phi(y=k|x)=\frac{exp[-d(f_\Phi(x),v^{(k)})]}{\sum_{k'=1}^{N_c} exp[-d(f_\Phi(x),v^{(k')})]}" />
</p>

<ins>**Step 4: classify queries**</ins>

The class with higher probability is the class assigned to the query image. 

<ins>**Step 5: compute the loss and backpropagate**</ins>

Only in training mode. Prototypical Networks use log-softmax loss, which is nothing but log over softmax loss. The log-softmax has the effect of heavily penalizing the model when it fails to predict the correct class, which is what we need.

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\large&space;J(\Phi)&space;=&space;-log(p_\Phi(y=k|x))\;of\;the\;true\;class\;k" title="\large J(\Phi) = -log(p_\Phi(y=k|x))\;of\;the\;true\;class\;k" />
</p>

<ins>**Pros and Cons of Prototypical Networks**</ins>

| Pros | Cons |
| --- | --- |
| Easy to understand | Lack of generalization |
| Very "visual" | Only use mean to decide prototypes, and ignore variance in support set |
| Noise resistant thanks to mean prototypes ||
| Can be adapted to Zero-shot setting ||

## II. The Omniglot Dataset

The Omniglot dataset is a benchmark dataset in Few-shot Learning. It contains 1,623 different handwritten characters from 50 different alphabets. 
The dataset can be found in [this repository](https://github.com/brendenlake/omniglot/tree/master/python). I used the `images_background.zip` and the `images_evaluation.zip` files.

## III. Implementation of ProtoNet for Omniglot

As suggested in the official paper, to increase the number of classes, **all the images are rotated by 90, 180 and 270 degrees**. Each rotation resulting in an additional class, so the total number of classes is now 6,492 (1,623 * 4). The training set contains images of 4,200 classes while the test set contains images of 2,292 classes.

The embedding part takes a (28x28x3) image and returns a column vector of length 64. The image2vector function is composed of **4 modules**. Each module consists of :
- A convolutional layer
- A batch normalization
- A ReLu activation function
- A 2x2 max pooling layer. 

![Embedding CNNs](https://github.com/cnielly/prototypical-networks-omniglot/blob/master/readme_images/embedding_CNN_1.jpg)

The chosen optimizer is **Adam**. The initial learning rate of 10<sup>−3</sup>, is cut in half at every epoch.

The model was trained on 5 epochs of 2,000 episodes each and was tested on 1,000 episodes. A new sample was randomly picked in the training set at each episode. 

<ins>**RESULTS**</ins>

I tried to reproduce the results of the paper.\
Training settings: 60 classes, 1 or 5 support points and 5 query points per class.\
Testing settings: 5-way and 20-way scenarios, same number of support and query points than during training.  

<table>
  <tr>
    <td></td>
    <td colspan="2" align="center">5-way</td>
    <td colspan="2" align="center">20-way</td>
  </tr>
  <tr>
    <td></td>
    <td>1-shot</td>
    <td>5-shot</td>
    <td>1-shot</td>
    <td>5-shot</td>
  </tr>
  <tr>
    <td>Obtained</td>
    <td>98.8%</td>
    <td>99.8%</td>
    <td>96.1%</td>
    <td>99.2%</td>
  </tr>
  <tr>
    <td>Paper</td>
    <td>98.8%</td>
    <td>99.7%</td>
    <td>96.0%</td>
    <td>98.9%</td>
  </tr>
</table>

I obtained similar results than the original paper, slightly better in some cases. This may be due to the sampling strategy which is not specified in the paper. I used random sampling at each episode. 
