"""Pre-process image datasets.

For each image dataset, we want to make one or more subsets suitable for a
binary classification (and calibration) task.

For ImageNet-1k, get data here https://huggingface.co/datasets/ILSVRC/imagenet-1k
(requires signing agreement and getting token)

AID can be downloaded here: https://huggingface.co/datasets/blanchon/AID
(e.g. via `git clone https://hf.co/datasets/blanchon/AID`, be sure to install git lfs)
Easiest way is to download it locally and use imagefolder loader.
(original download links from https://captain-whu.github.io/AID/ are stale)

Places365 can be downloaded here: http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar

1. Load dataset/split as huggingface dataset
2. Choose a target class
3. Collect all target class images
4. For each target class image, sample k non-target images (k=10 by default)
5. Resize all images to 224x224
6. Save the new dataset to disk

# "Llama" is ImageNet class 355
python dataprep.py imagenet --split train 355 ./data/llama-train
python dataprep.py imagenet --split validation 355 ./data/llama-test
# "Golden Retriever" is ImageNet class 207
python dataprep.py imagenet --split train 207 ./data/golden-retriever-train
python dataprep.py imagenet --split validation 207 ./data/golden-retriever-test
# "Labrador Retriever" is ImageNet class 208
python dataprep.py imagenet --split train 208 ./data/labrador-retriever-train
python dataprep.py imagenet --split validation 208 ./data/labrador-retriever-test
# "Crane" (bird, not machine) is ImageNet class 134
python dataprep.py imagenet --split train 134 ./data/crane-train
python dataprep.py imagenet --split validation 134 ./data/crane-test
# "Sunglasses" is ImageNet class 837
python dataprep.py imagenet --split train 837 ./data/sunglasses-train
python dataprep.py imagenet --split validation 837 ./data/sunglasses-test
# "Beach" is ImageNet class 978, AID class 3
python dataprep.py imagenet --split train 978 ./data/beach-train
python dataprep.py imagenet --split validation 978 ./data/beach-test
python dataprep.py aid --datadir /path/to/AID 3 ./data/beach-test-ood
"""

import argparse
import inspect
import io
import logging
import logging.config
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
    aid_dataset = filter_dataset(aid_dataset, AID_LABELS, transform)
    aid_dataset.save_to_disk(args.outdir / "aid")

    places365_val_dataset = datasets.load_dataset(
        "imagefolder",
        data_dir=args.datadir / "places365_standard",
        split="val",
    )
    places365_val_dataset = filter_dataset(places365_val_dataset, PLACES365_LABELS, transform, AID_LABELS)
    places365_val_dataset.save_to_disk(args.outdir / "places365-val")

    LOG.warning(
        "Places365 dataset is large and slow to process (~20 minutes w/SSD and 4 cores)"
    )
    places365_train_dataset = datasets.load_dataset(
        "imagefolder",
        data_dir=args.datadir / "places365_standard",
        split="train",
    )
    places365_train_dataset = filter_dataset(places365_train_dataset, PLACES365_LABELS, transform, AID_LABELS)
    places365_train_dataset.save_to_disk(args.outdir / "places365-train")

    # imagenet_dataset = datasets.load_dataset(
    #     "imagenet-1k",
    #     trust_remote_code=True,
    #     split=args.split,
    # )
    total_time = timer() - start_time
    LOG.info(f"Ran the script in {total_time:.3f} seconds")


def filter_dataset(dataset, label_names, transform, new_label_names=None):
    """Filter the dataset to include only the specified labels."""
    label_nums = tuple(
        dataset.features["label"].str2int(label) for label in label_names
    )
    label_map = {orig: i for i, orig in enumerate(label_nums)}
    dataset = dataset.filter(
        lambda examples: [label in label_nums for label in examples["label"]],
        batched=True,
        num_proc=4,
        batch_size=1000,
    )
    dataset = dataset.map(
        lambda x: {
            "image": transform(x["image"]),
            "label": label_map[x["label"]],
        },
        num_proc=4,
    )
    if new_label_names is not None:
        label_names = new_label_names
    dataset = dataset.cast_column("label", datasets.ClassLabel(names=label_names))
    return dataset


def convert_to_jpeg(image, quality: int = 85):
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
        "datadir",
        type=pathlib.Path,
        default=None,
        help="directory containing the datasets",
    )
    parser.add_argument(
        "outdir",
        type=pathlib.Path,
        help="directory to save the processed dataset",
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
