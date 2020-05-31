# Prototypical Networks on the Omniglot Dataset
An implementation of "Prototypical Networks for Few-shot Learning" on a notebook in Pytorch

## I. Prototypical Networks

**Prototypical Networks** were introduced by Snell et al. in 2017 (https://arxiv.org/abs/1703.05175). 
They started from a pre-existing architecture called **Matching Networks** introduced in a previous paper (https://arxiv.org/abs/1606.04080).
They are both part of a broader familly of algorithms called **Metric Learning Algorithms**, 
and the success of these networks is based on their capacity to understand the similarity relationship among samples.

"Our approach, prototypical networks, is based on the idea that there exists an embedding in which points cluster around a single prototype 
representation for each class." (from the original paper *Prototypical Networks for Few-shot Learning*) 

In other words, there exists a mathematical representation of the images, called the **embedding space**, 
in which images of the same class gather in clusters. 
The main advantage of working in that space is that two images that look the same will be close to each other, 
and two images that are completely different will be far away from each other. 

[Image of the clusters in the embedding space]

Here “close” is linked to a distance metric that needs to be defined. We usually take the cosine distance of the Euclidean distance.  

Unlike typical deep learning architecture, prototypical networks do not classify the image directly, and instead learn the mapping of an image in the metric space. 
To do so, the algorithm does several “loops” called **episodes**. Each episode is designed to mimic the few-shot task. Let’s describe in details one episode in training mode:

<ins>**Notations:**</ins>

In Few-shot classification, we are given a dataset with few images per class. N<sub>c</sub> classes are randomly picked, and for each class we have two sets of images: the support set (size N<sub>c</sub>) and the query set (size N<sub>q</sub>). 

[Image of the matrix representation: one line = one classe, Ns columns of support images, Nq of query images]

<ins>**Step 1: embed the images**</ins>

First, we need to transform the images into vectors. This step is called the embedding, and is performed thanks to an "Image2Vector" model, which is a Convolutional Neural Network (CNN) based architecture.

<ins>**Step 2: compute class prototypes**</ins>

This step is similar to K-means clustering (unsupervised learning) were a cluster is represented by its centroid. 
The embeddings of the support set images are averaged to form a class prototype.

[LateX formula v_k = sum(x_i in Sk, f_theta(x_i)]

S<sub>k</sub> denotes the set of examples labeled with class k.

The prototype of a class can be seen as the representative of the class. 

<ins>**Step 3: compute distance between queries and prototypes**</ins>

This step consists in classifying the query images. To do so, we compute the distance between the images and the prototypes. Metric choice is crucial, and the inventors of Prototypical Networks must be credited to their choice of distance metric. They noticed that their algorithm and Matching Networks both perform better using Euclidean distance than when using cosine distance. 

[Put formulas of cosine and euclidean distances.]

Once distances are computed, a softmax is performed over distances to the prototypes in the embedding space, to get probabilities. 

[Put formula of probabilty]

<ins>**Step 4: classify queries**</ins>

The class with higher probability is the class assigned to the query image. 

<ins>**Step 5: compute the loss and backpropagate**</ins>

Only in training mode. Prototypical Networks use log-softmax loss, which is nothing but log over softmax loss. The log-softmax has the effect of heavily penalizing the model when it fails to predict the correct class, which is what we need.

J(φ) = − log pφ(y = k | x) of the true class k

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

The embedding part takes a (28x28x3) image and returns a column vector of length 64. The image2vector function is composed of **4 convolutional blocks**. As shown in the above image, each block consists of :
- A CNN layer 
- A batch normalization
- A ReLu activation function
- A 2x2 max pooling layer. 

[Image showing the architecture of the convolutional block]


The chosen optimizer is **Adam**. The initial learning rate of 10<sup>−3</sup>, is cut in half at every epoch.

The model was trained on 5 epochs of 2,000 episods each, and was tested on 1,000 episodes. 

<ins>**RESULTS**</ins>

I tried to reproduced the results of the paper, and this is what I obtained: 




