# CS475 Fall 2024 ML Class Project -- Calibrating Zero-Shot Image Classifiers

This class project investigates the effectiveness of calibration methods for zero-shot binary image classifiers and whether they transfer across classes and domains.

## Team Members

- Ayush Agarwal (aagarw41@jhu.edu)
- Kahnran Braxton (kbraxto6@jhu.edu)
- Cameron Carpenter (ccarpe18@jhu.edu)
- William Shiber (wshiber1@jhu.edu)

## Quickstart

The code in this repository is designed to be run in Google Colab.

Alternatively, to run locally, create a Python virtual environment and run:

```bash
pip install -r requirements.txt
```

## Data Processing

If running this project in Google Colab, the preprocessed data is already
set up to be loaded into the notebook from Google Drive shared links.

If running locally, use the following instructions to fetch the data and
process it for use in the notebook:

```bash
mkdir data/raw
pushd data/raw
# note: AID project page is https://captain-whu.github.io/AID/ but links are stale
# note: copy of the dataset available at https://huggingface.co/datasets/blanchon/AID
# note: dataset is about 3GB but requires about 6GB for intermediate processing
# note: ensure git-lfs is installed first, e.g. with `sudo apt install git-lfs`
git lfs install
git clone https://hf.co/datasets/blanchon/AID
rm -rf AID/.git  # not requried but frees up space
# note: places365 project page is at http://places2.csail.mit.edu/
# note: download links are at http://places2.csail.mit.edu/download-private.html
# note: places365 is about 25GB and needs another 25GB for intermediate processing
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
tar -xvf places365standard_easyformat.tar
rm places365standard_easyformat.tar  # not required but frees up space
popd
python dataprep.py ./data/raw/ ./data/processed
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