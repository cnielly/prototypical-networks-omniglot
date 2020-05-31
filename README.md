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

<ins>**Step 3: compute distance between queries and prototypes **</ins>

This step consists in classifying the query images. To do so, we compute the distance between the images and the prototypes. Metric choice is crucial, and the inventors of Prototypical Networks must be credited to their choice of distance metric. They noticed that their algorithm and Matching Networks both perform better using Euclidean distance than when using cosine distance. 

[Put formulas of cosine and euclidean distances.]

Once distances are computed, a softmax is performed to get a probability on each component, and a sum equal to 1.

<ins>**Step 4: classify queries **</ins>

<ins>**Step 5: compute the loss and backpropagate **</ins>

Only in training mode. 

## II. The Omniglot Dataset

## III. Implementation of ProtoNet for Omniglot
