# Assignment 2: Building a Complete Visual Perception Pipeline

This directory contains the complete implementation, experiments, and analysis for **Assignment-2** of **DA6401 - Introduction to Deep Learning**.

The focus of this assignment is to transition to PyTorch to build a comprehensive, multi-stage Visual Perception Pipeline. Rather than solving isolated problems, the final deliverable is a cohesive system capable of detecting, classifying, and segmenting subjects within an image using the **Oxford-IIIT Pet Dataset**.

For the public Weights & Biases report detailing the analysis of this assignment, click here: [W\&B Report Link](https://wandb.ai/cs25m050-indian-institute-of-technology-madras/DA6401_Assignment_02/reports/DA6401-Assignment-2-Report--VmlldzoxNjM1NTA5NA?accessToken=l1g26d64450a2ucaehw61b8gd4i932u2pe6hssvhufvq99s8k1lk0i5zjm3ft197)

For the codebase of this current project, click here: [GitHub Link](https://github.com/Vishnuteja-Surla/Multitask-Visual-Pipeline) *(Note: Follows the official Assignment-2 GitHub Skeleton)*

-----

## Assignment Objectives

The objectives of this assignment are to:

  * **Task 1:** Implement a VGG11 classification architecture from scratch using standard `torch.nn` modules to classify 37 pet breeds.
  * **Task 2:** Extend the model with an Encoder-Decoder for single-object localization via bounding box regression.
  * **Task 3:** Construct a U-Net style semantic segmentation network using the VGG11 encoder and a structurally mirroring symmetric decoder.
  * **Task 4:** Integrate all components into a Unified Multi-Task Pipeline featuring a single forward pass.

-----

## Constraints & Academic Integrity

  * **Frameworks:** Transitioning strictly to PyTorch for this pipeline.
  * **Permitted libraries:** `torch`, `numpy`, `matplotlib`, `scikit-learn`, `wandb`, `albumentations`.
  * **AI Usage:** Tools like ChatGPT or Claude are permitted only as conceptual aids; they must not be used to generate the final code submission.
  * **Collaboration:** This is an individual assignment; collaborations are strictly prohibited.
  * **Data Integrity:** Training and test datasets must be strictly isolated to prevent data leakage, and any attempt to artificially inflate accuracy will result in a zero.
  * **Plagiarism:** All submissions will undergo rigorous plagiarism and AI-generated code detection.

-----

## Implementation & Architecture Notes

Instead of a configurable CLI, this multi-task pipeline is evaluated based on strict architectural and functional requirements:

  * **Custom Modules (No Built-ins allowed)**
      * A custom Dropout layer inheriting from `torch.nn.Module`, implementing inverted dropout scaling and deterministic behavior during evaluation.
      * A custom Intersection over Union (IoU) loss function.
  * **Architectural Rules**
      * Pre-built VGG models are strictly prohibited.
      * Learnable upsampling in the U-Net must use Transposed Convolutions; standard interpolation algorithms are not permitted.
      * Feature fusion must occur by concatenating upsampled feature maps with spatially-aligned encoder maps.
  * **Unified Forward Pass**
      * A single `forward(self, x)` method must simultaneously yield:
          * **Breed Label:** 37-class classification logits.
          * **Bounding Box:** 4 continuous coordinate values `[Xcenter, Ycenter, width, height]`.
          * **Segmentation Mask:** Dense, pixel-wise spatial map.
  * **Automated Evaluation Metrics**
      * Classification: Macro $F_{1}$-Score.
      * Detection: Mean Average Precision (mAP).
      * Segmentation: Dice Similarity Coefficient.

-----

## Weights & Biases Report

A **public Weights & Biases report** accompanies this assignment and includes:

  * **Regularization & Dynamics:** Analysis of the effect of Batch Normalization and the custom Dropout implementation on the generalization gap and activation distributions.
  * **Transfer Learning Showdown:** Empirical comparison between a Strict Feature Extractor, Partial Fine-Tuning, and Full Fine-Tuning strategies on the segmentation pipeline.
  * **Black Box Visualization:** Extracted feature maps from early and late convolutional layers to observe the transition from localized edges to high-level semantic shapes.
  * **Object Detection Analysis:** Bounding box predictions overlaid on test images with Confidence Scores, IoU calculations, and failure case identification.
  * **Segmentation Evaluation:** Mathematical explanation of why the Dice Coefficient is superior to Pixel Accuracy for imbalanced segmentation tasks.
  * **Final Pipeline Showcase:** Evaluation of the pipeline on 3 novel "in-the-wild" pet images downloaded from the internet.
  * **Meta-Analysis:** Retrospective reflection on architectural decisions, loss formulations, and potential task interference within the unified backbone.

-----

## Submission Notes

  * A Public W\&B Report link must be accessible during evaluation to avoid a negative marking penalty.
  * The formal submission of the codebase and report must be completed via Gradescope.
  * No extensions will be granted beyond the provided deadline under any circumstances.

-----

## Timeline

  * **Release Date:** 21th March 2026, 00:00 AM
  * **Submission Deadline:** 12th April 2026, 23:59 PM
  * **Late Submission Deadline:** 14th April 2026, 23:59 PM (with penalty)
  * **Submission Platform:** Gradescope

-----

## Relation to Course Objectives

This assignment directly supports the course goals of:

  * Transitioning from manual mathematical implementations to standard deep learning frameworks (PyTorch).
  * Understanding and building complex, modern architectures like VGG and U-Net.
  * Designing multi-task learning systems that share representations across varied objectives.
  * Evaluating complex spatial models using domain-specific metrics (mAP, Dice Score, IoU).