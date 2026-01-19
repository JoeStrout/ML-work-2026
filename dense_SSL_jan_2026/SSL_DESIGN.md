# Self-Supervised Learning of Dense Classification in EM Images

## Overview

This project proposed to tackle dense semantic labeling of 3D EM images using a self-supervised learning (SSL) approach similar to LeJEPA, but operating at the level of pixels rather than entire images.  The approach involves:

1. An encoder/classifier network, that produces a low-dimensional output for every pixel in the central region of a 2D image (i.e., we will trim away the border area where insufficient data would exist to fill the receptive field of the corresponding outputs).  The per-pixel output may be something like a 20-dimensional softmaxed vector that could be considered an (unsupervised) class label.

2. A prediction network that learns the following pretext task: given the outputs (from network 1) for four surrounding Z-slices (2 before and 2 after), predict the outputs for the central slice.  This is analogous to the CBOW task in NLP.

3. A regularization term that ensures that the outputs from network 1 do not collapse to a trivial solution.

The loss function would depend on the regularization term, as well as the difference between prediction and actual outputs from network 2.  The big idea is to cause network 1 to learn classification vectors that are _useful_ for the prediction task, and therefore encode semantic classes that will also be useful for downstream tasks.

## Output Representation

I am starting with the assumption that the best format for our output vectors is (1) low dimensional, and (2) softmaxed.  My concern is that if we use high-dimensional or arbitrary embedding vectors, these will simply transfer the input image directly, putting all the burden of solving the pretext task on the prediction network, which is not where we want it.

Questions:
- Is this actually necessary, or would a standard (not softmaxed) encoding vector actually work as well or better?
  - ChatGPT thinks the pass-through concern is valid; logits -> softmax is a good idea.
- Does a 1-hot category vector even solve the problem, or will the encoder network simply reduce pixel intensity to 20 values (or otherwise fail to encode semantic information)?
  - It might without proper regularization; with it, intensity quantization becomes suboptimal.
- How many dimensions (K) should our output vectors have?
  - We should probably over-cluster (e.g. K=50) to avoid early semantic bottlenecks.

## Encoder/Classifier Network

This network takes as input a 2D grayscale image, and outputs a D-dimensional vector for every pixel within a central region.

Questions:
- What network architectures are appropriate for this task?
  - In general: a U-Net/U-Net++, ConvNeXt with FPN-style head, or even a lightweight 2D CNN with large receptive field.  ViT/hybrid is also possible.
- Can we use off-the-shelf models from timm for this?
  - YES! Pull down a timm backbone (ConvNeXt, ResNet, ViT-hybrid), remove the classifier, and attach a 1x1 conv head producing K channels; upsample/align to pixel grid; and softmax over K.

## Regularization

We need to avoid trivial solutions where the encoder outputs collapse to a single value.  At the same time, we don't want to force equipartition over categories, as in real images, the likely semantic categories vary widely in their pixel density (e.g. cytoplasm vs. membrane).  And there will be some categories that will be heavily used in some images, but not at all in others (e.g. nucleoplasm).

- What regularization function can ensure a reasonable distribution of categories?

See ChatGPT's answer here:
https://chatgpt.com/c/696e50af-d1f4-8326-b3a8-2f9e33bd0818

## Prediction Network

The job of this network is to take four dense outputs (from the encoder/classifier network), and predict the fifth.  This is largely a matter of perceiving and applying the "flow" as we move through Z.  But it is hoped that to do this effectively will require some understanding of various cellular structures and their 3D shape (i.e., how their 2D shape tends to vary with Z).

We want to keep the predictor small/simple to ensure semantics live in the encoder.

Questions:
- What network architectures are appropriate for this task?
  - Small U-Net over stacked inputs (4×K channels → K)
  - Shallow ConvNet with dilations
  - JEPA-style predictor: few conv blocks + MLP head
- Are there any off-the-shelf models in timm that might apply?
  - Not directly, though we might be able to use some timm backbones as components.

