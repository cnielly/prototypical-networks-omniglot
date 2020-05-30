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

Here “close” is linked to a distance metric that needs to be defined. We usually take the cosine distance of the Euclidean distance.  

## II. The Omniglot Dataset

## III. Implementation of ProtoNet for Omniglot
