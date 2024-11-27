# CS475 Fall 2024 ML Class Project -- Calibrating Zero-Shot Image Classifiers

This class project investigates the effectiveness of calibration methods for zero-shot binary image classifiers and whether they transfer across classes and domains.

## Team Members

- Ayush Agarwal (aagarw41@jhu.edu)
- Kahnran Braxton (kbraxto6@jhu.edu)
- Cameron Carpenter (ccarpe18@jhu.edu)
- William Shiber (wshiber1@jhu.edu)

## Quickstart

The code in this repository is designed to be run in Google Colab.

## Experiments

### Calibration methods

- Platt Scaling
- Isotonic Regression
- Similarity Binning

### Conditions

- Same class, same domain
- Different class, same domain
- Different domain, same class
- Different domain, different class

### Datasets

  - Imagenet (selected classes)
  - AID (selected classes)


### Results

- Classifier results
  - Confusion matrix
  - Accuracy, precision, recall, F1
- Calibration results
  - Reliability plots
  - smECE
  - Brier score