# CS475 Fall 2024 ML Class Project -- Calibrating Zero-Shot Image Classifiers

This class project investigates the effectiveness of calibration methods for zero-shot binary image classifiers and whether they transfer across classes and domains.

## Team Members

- Ayush Agarwal (aagarw41@jhu.edu)
- Kahnran Braxton (kbraxto6@jhu.edu)
- Cameron Carpenter (ccarpe18@jhu.edu)
- William Shiber (wshiber1@jhu.edu)

## Quickstart

The code in this repository is designed to be run in Google Colab.

## Data Processing

If running this project in Google Colab, the preprocessed data is already
set up to be loaded into the notebook from Google Drive.

If running locally, use the following instructions to fetch the data and
process it for use in the notebook:

```bash
mkdir data/raw
pushd data/raw
# note: ensure git-lfs is installed first, e.g. with `sudo apt install git-lfs`
git lfs install
git clone https://hf.co/datasets/blanchon/AID
# note: places365 is about 25GB and needs another 25GB for intermediate processing
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
tar -xvf places365standard_easyformat.tar
rm places365standard_easyformat.tar
popd
```

## Experiments

### Calibration methods

- Platt Scaling
- Isotonic Regression
- Similarity Binning Averaging

### Conditions

- Same class, same domain
- Different class, same domain
- Different domain, same class
- Different domain, different class

### Datasets

  - Imagenet (selected classes)
  - Places365 (selected classes)
  - AID (selected classes)



### Results

- Classifier results
  - Confusion matrix
  - Accuracy, precision, recall, F1
- Calibration results
  - Reliability plots
  - smECE
  - Brier score