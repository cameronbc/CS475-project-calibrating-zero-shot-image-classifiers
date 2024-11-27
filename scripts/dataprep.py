"""Pre-process image datasets.

For each image dataset, we want to make one or more subsets suitable for a
binary classification (and calibration) task.

For ImageNet-1k, get data here https://huggingface.co/datasets/ILSVRC/imagenet-1k
(requires signing agreement and getting token)

AID can be downloaded here: https://huggingface.co/datasets/blanchon/AID
Easiest way is to download it locally and use imagefolder loader
(original download links from https://captain-whu.github.io/AID/ are stale)

1. Load dataset/split as huggingface dataset
2. Choose a target class
3. Collect all target class images
4. For each target class image, sample k non-target images (k=10 by default)
5. Resize all images to 224x224
6. Save the new dataset to disk

# "Llama" is ImageNet class 355
python dataprep.py imagenet --split train 355 hf-data/llama-train
python dataprep.py imagenet --split validation 355 hf-data/llama-test
# "Golden Retriever" is ImageNet class 207
python dataprep.py imagenet --split train 207 hf-data/golden-retriever-train
python dataprep.py imagenet --split validation 207 hf-data/golden-retriever-test
# "Labrador Retriever" is ImageNet class 208
python dataprep.py imagenet --split train 208 hf-data/labrador-retriever-train
python dataprep.py imagenet --split validation 208 hf-data/labrador-retriever-test
# "Crane" (bird, not machine) is ImageNet class 134
python dataprep.py imagenet --split train 134 hf-data/crane-train
python dataprep.py imagenet --split validation 134 hf-data/crane-test
# "Sunglasses" is ImageNet class 837
python dataprep.py imagenet --split train 837 hf-data/sunglasses-train
python dataprep.py imagenet --split validation 837 hf-data/sunglasses-test
# "Beach" is ImageNet class 978, AID class 3
python dataprep.py imagenet --split train 978 hf-data/beach-train
python dataprep.py imagenet --split validation 978 hf-data/beach-test
python dataprep.py aid --datadir /path/to/AID 3 hf-data/beach-test-ood
"""

import argparse
import inspect
import logging
import logging.config
import sys
import pathlib
from timeit import default_timer as timer
from typing import Optional

import datasets
import torchvision.transforms

MODULE_NAME = pathlib.Path(__file__).resolve().stem
LOG = logging.getLogger(MODULE_NAME)


def main(args: Optional[argparse.Namespace] = None) -> Optional[int]:
    """Execute the command with the given arguments."""
    if not args:
        parser = build_parser()
        args = parser.parse_args()
    configure_logging(args.verbose)
    start_time = timer()
    if args.dataset == "imagenet":
        dataset = datasets.load_dataset(
            "imagenet-1k",
            trust_remote_code=True,
            split=args.split,
        )
    elif args.dataset == "aid":
        dataset = datasets.load_dataset(
            "imagefolder",
            data_dir=args.datadir,
            split="train",  # AID has no splits
        )
    else:
        LOG.error(f"Unknown dataset: {args.dataset}")
        return 1

    # targets = dataset.filter(lambda x: x["label"] == args.target, num_proc=4)
    targets = dataset.filter(
        lambda examples: [label == args.target for label in examples["label"]],
        batched=True,
        num_proc=4,
        batch_size=1000,
    )
    original_num_target = len(targets)
    if args.num_target and args.num_target < original_num_target:
        targets = targets.shuffle(seed=42).select(range(args.num_target))
    LOG.info(f"Selected {len(targets)} target class images from {original_num_target}")
    non_targets = (
        dataset.filter(
            lambda examples: [label != args.target for label in examples["label"]],
            batched=True,
            num_proc=4,
            batch_size=1000,
        )
        .shuffle(seed=42)
        .select(range(args.num_non_target * len(targets)))
    )
    dataset = datasets.concatenate_datasets([targets, non_targets])
    resizer = torchvision.transforms.Resize(
        size=(224, 224),
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        max_size=None,
        antialias=True,
    )
    dataset = dataset.map(
        lambda x: {
            "image": resizer(x["image"]).convert("RGB"),
            "label": x["label"],
        },
        num_proc=4,
    )
    dataset.save_to_disk(args.outdir)
    total_time = timer() - start_time
    LOG.info(f"Ran the script in {total_time:.3f} seconds")


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
        "--datadir",
        "-d",
        type=pathlib.Path,
        default=None,
        help="directory containing the dataset",
    )
    parser.add_argument(
        "--num-target",
        "-n",
        type=int,
        default=None,
        help="number of target class images to sample",
    )
    parser.add_argument(
        "--num-non-target",
        "-m",
        type=int,
        default=10,
        help="number of non-target class images to sample per target image",
    )
    parser.add_argument(
        "--split",
        "-s",
        default=None,
        help="dataset split to load [default=None]",
    )
    parser.add_argument(
        "dataset",
        choices=("imagenet", "aid"),
        help="dataset name to process",
    )
    parser.add_argument(
        "target",
        type=int,
        help="target class for binary classification",
    )
    parser.add_argument(
        "outdir",
        type=pathlib.Path,
        help="directory to save the processed dataset",
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
