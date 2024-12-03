"""Pre-process Places365 and AID image datasets for upload to Google Drive.

- AID (Aerial Image Dataset)
  - Xia, Gui-Song, et al. "AID: A benchmark data set for performance evaluation of aerial scene classification." IEEE Transactions on Geoscience and Remote Sensing 55.7 (2017): 3965-3981.
  - project page: https://captain-whu.github.io/AID/
  - download instructions: install git lfs, then `git clone https://hf.co/datasets/blanchon/AID`
- Places365
  - Zhou, Bolei, et al. "Places: A 10 million image database for scene recognition." IEEE transactions on pattern analysis and machine intelligence 40.6 (2017): 1452-1464.
  - project page: http://places2.csail.mit.edu/
  - downloads page: http://places2.csail.mit.edu/download-private.html
  - download instructions: wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar

This script does the following:

1. Loads dataset train splits as huggingface datasets
2. Filters down to 20 classes that are shared between AID and Places365
3. Map AID class names to Places365 class names
4. Process all images to 224x224 RGB JPEGs
5. Save the new datasets to disk as huggingface datasets

The resulting datasets can then be zipped and uploaded to Google Drive for use in Google Colab.
"""

import argparse
import inspect
import io
import logging
import logging.config
import shutil
import sys
import pathlib
from timeit import default_timer as timer
from typing import Optional

import datasets
import PIL.Image
import torchvision.transforms


MODULE_NAME = pathlib.Path(__file__).resolve().stem
LOG = logging.getLogger(MODULE_NAME)

AID_LABELS = (
    "Airport",
    "BaseballField",
    "Beach",
    "Bridge",
    "Church",
    "Desert",
    "Farmland",
    "Forest",
    "Industrial",
    "Meadow",
    "MediumResidential",
    "Mountain",
    "Park",
    "Parking",
    "Playground",
    "Pond",
    "RailwayStation",
    "River",
    "School",
    "Viaduct",
)

PLACES365_LABELS = (
    "airfield",
    "stadium-baseball",
    "beach",
    "bridge",
    "church-outdoor",
    "desert-sand",
    "farm",
    "forest-broadleaf",
    "industrial_area",
    "field-cultivated",
    "residential_neighborhood",
    "mountain",
    "park",
    "parking_garage-outdoor",
    "playground",
    "pond",
    "train_station-platform",
    "river",
    "schoolhouse",
    "viaduct",
)


def main(args: Optional[argparse.Namespace] = None) -> Optional[int]:
    """Execute the command with the given arguments."""
    if not args:
        parser = build_parser()
        args = parser.parse_args()
    configure_logging(args.verbose)
    start_time = timer()

    resizer = torchvision.transforms.Resize(
        size=(224, 224),
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        max_size=None,
        antialias=True,
    )
    transform = lambda img: convert_to_jpeg(resizer(img))

    aid_dataset = datasets.load_dataset(
        "imagefolder",
        # note: HF puts the data in a subdirectory named "data"
        data_dir=args.datadir / "AID" / "data",
        split="train",  # AID has no splits, train is default
    )
    aid_dataset = filter_dataset(aid_dataset, AID_LABELS, transform, PLACES365_LABELS)
    aid_path = args.outdir / "aid"
    aid_dataset.save_to_disk(aid_path)
    if not args.no_zip:
        LOG.info("Zipping AID dataset")
        shutil.make_archive(aid_path, "zip", aid_path)

    LOG.warning(
        "Places365 dataset is large and slow to process (~10 minutes w/SSD and 4 cores)"
    )
    places365_dataset = datasets.load_dataset(
        "imagefolder",
        data_files={
            "train": str(args.datadir / "places365_standard" / "train" / "**"),
            "validation": str(args.datadir / "places365_standard" / "val" / "**"),
        },
    )
    if args.places_val:
        places365_val_dataset = filter_dataset(
            places365_dataset["validation"],
            PLACES365_LABELS,
            transform,
        )
        places365_val_path = args.outdir / "places365-val"
        places365_val_dataset.save_to_disk(places365_val_path)
        if not args.no_zip:
            LOG.info("Zipping Places365 validation dataset")
            shutil.make_archive(places365_val_path, "zip", places365_val_path)
    places365_train_dataset = filter_dataset(
        places365_dataset["train"],
        PLACES365_LABELS,
        transform,
    )
    places365_train_path = args.outdir / "places365"
    places365_train_dataset.save_to_disk(places365_train_path)
    if not args.no_zip:
        LOG.info("Zipping Places365 train dataset")
        shutil.make_archive(places365_train_path, "zip", places365_train_path)

    total_time = timer() - start_time
    LOG.info(f"Ran the script in {total_time:.3f} seconds")


def filter_dataset(dataset, label_names, transform, new_label_names=None):
    """Filter the dataset to include only the specified labels."""
    label_nums = tuple(
        dataset.features["label"].str2int(label) for label in label_names
    )
    label_map = {orig: i for i, orig in enumerate(label_nums)}
    dataset = dataset.filter(
        lambda labels: [label in label_nums for label in labels],
        input_columns=["label"],
        batched=True,
        batch_size=1024,
    )
    dataset = dataset.map(
        lambda example: {
            "image": transform(example["image"]),
            "label": label_map[example["label"]],
        },
        num_proc=4,
    )
    if new_label_names is not None:
        label_names = new_label_names
    dataset = dataset.cast_column("label", datasets.ClassLabel(names=label_names))
    return dataset


def convert_to_jpeg(image, quality: int = 90):
    """Convert a PIL image to JPEG format with specified quality."""
    buffer = io.BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return PIL.Image.open(buffer)


def configure_logging(verbosity: int = 0) -> None:
    """Set the log level based on the desired verbosity."""
    log_level = logging.WARNING
    if verbosity == 1:
        log_level = logging.INFO
    if verbosity >= 2:
        log_level = logging.DEBUG
    logging.config.dictConfig(
        {
            "version": 1,
            "formatters": {"brief": {"format": "%(levelname)s:%(name)s:%(message)s"}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "brief",
                }
            },
            "loggers": {MODULE_NAME: {"level": log_level, "handlers": ["console"]}},
        }
    )


def build_parser(
    description: str = inspect.cleandoc(__doc__),
) -> argparse.ArgumentParser:
    """Create a command line arguments parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=description,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        default=0,
        action="count",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--places-val",
        action="store_true",
        help="if flag is given, process Places365 validation set in addition to train set",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="if flag is given, skip writing zip archives",
    )
    parser.add_argument(
        "datadir",
        type=pathlib.Path,
        default=None,
        help="directory containing the raw datasets to process",
    )
    parser.add_argument(
        "outdir",
        type=pathlib.Path,
        help="directory to save the processed dataset",
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
