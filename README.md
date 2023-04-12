# CanineMammaryTumors

This repository contains code examples for working with a dataset of microscopic whole slide images.
It is an adaptation of the CanineCutaneousTumors repository that can be found [here](https://github.com/DeepPathology/CanineCutaneousTumors).

<p float="left">
  <img src="CMT_annotation_SlideRunner.png" width="500" alt="Dataset"/>
</p>

## Segmentation
For the segmentation network training of different classes a Jupyter notebook is provided.

* Training: [segmentation_training_pipeline.ipynb](segmentation/segmentation_training_pipeline.ipynb)
* Inference on Patch Level: [patch_segmentation_inference.ipynb](segmentation/patch_segmentation_inference.ipynb)
* Inference on Slide Level: [slide_segmentation_inference.ipynb](segmentation/slide_segmentation_inference.ipynb)



## Visualization Patch Prediction Tubule Formation

<p float="left">
  <img src="single_patch_pred_tub_1_small.png" width="300" alt="Segmentation prediction"/>
</p>

## Visualization Patch Prediction Nuclear Pleomorphism

<p float="left">
  <img src="single_patch_pred_small.png" width="300" alt="Segmentation prediction"/>
</p>

## Limitations 
Due to the limitation of having only a small dataset of such complex images, the segmentation accuracy was low. 
However, this pipeline can be used as a baseline for future works.


<!-- ## Pre-trained models
We provide two pre-trained models for patch segmentation and classification. These can be found in the 
[models](models) folder. 
<!-- A detailed evaluation of these models can be found in our ScientificData paper:   
> Link to paper --> -->



  