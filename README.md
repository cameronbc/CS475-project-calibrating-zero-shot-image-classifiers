# CS475 Fall 2024 ML Class Project -- Calibrating Zero-Shot Image Classifiers

This class project investigates the effectiveness of calibration methods for zero-shot binary image classifiers and whether they transfer across classes and domains.

## Team Members

- Ayush Agarwal (aagarw41@jhu.edu)
- Kahnrad Braxton (kbraxto6@jhu.edu)
- Cameron Carpenter (ccarpe18@jhu.edu)
- William Shiber (wshiber1@jhu.edu)

## Quickstart

The code in this repository is designed to be run in Google Colab. The data has already been prepared and uploaded to Google Drive for use there.

## Local Execution

Alternatively, to run locally, create a Python virtual environment and run:

```bash
pip install -r requirements.txt
```

### Data Processing

If running this project in Google Colab, the preprocessed data is already set up to be loaded into the notebook from Google Drive shared links. The data can be downloaded locally from these links:

- [cleaned Places365 subset](https://drive.google.com/file/d/1w-0LncVMfBsdtqX7jT-jCTdAnLBZuFtU/view?usp=drive_link)
- [cleand AID subset](https://drive.google.com/file/d/1CTdCFoo88_ygMb2PNGnK3QTi8xk_naoA/view?usp=drive_link)

To re-create the cleaned data subsets from the original data sources, follow these instructions for fetching the data and running the `dataprep.py` script locally after activating the python envirnoment:

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
