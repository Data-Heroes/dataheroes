import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import pandas as pd


from pycocotools.coco import COCO
from tqdm import tqdm
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision import models
from typing import Optional
from sklearn.decomposition import PCA

from PIL import Image
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root-dir",
    required=True,
    help="Absolute or relative path to the root directory of the COCO dataset.",
)
parser.add_argument(
    "--export-dir",
    required=True,
    help="Absolute or relative path to the root directory where the parsed data will be saved.",
)
parser.add_argument(
    "--adjust-bbox-ratio",
    default=0.,
    help="Ratio between [0, 1] that resizes every bounding box for more context.",
)
parser.add_argument(
    "--reduce-dimensions-to",
    default=256,
    help="Integer value used to reduce the size of the original embedding to a given size. These embeddings are saved along with the original ones.",
)
parser.add_argument(
    "--debug",
    default=0.,
    choices=[0, 1],
    help="In debugging mode, we can visualize additional information to understand the process.",
)

# COCO split metadata dictionary.
split_info = {
    "train": {
        "annotations_file": "instances_train2017.json"
    },
    "val": {
        "annotations_file": "instances_val2017.json"
    }
}


def export(split: str, root_dir: Path, export_dir: Path, num_max_samples: Optional[int] = None, debug: bool = False, adjust_bbox_ratio: float = 0.0):
    """Function used to compute the embeddings of COCO cropped bounding boxes from their afferent images and save them to disk.

    Args:
        split (str): COCO split, choices: ('train', 'val')
        root_dir (Path): root directory of COCO data
        export_dir (Path): root directory where the parsed data will be saved
        num_max_samples (Optional[int], optional): parameter used to limit the maximum number of exports. If None, all the samples will be exported.
        debug (bool, optional): if true, some visualizations will be shown to debug the bounding boxes. Defaults to False.
        adjust_bbox_ratio (float, optional): ratio used to increase the bounding box size to add additional context. Defaults to 0.0.
    """

    assert split in ("train", "val"), f"split: '{split}' not supported. Please choose between ('train', 'val')."
    assert 0. <= adjust_bbox_ratio <= 1., "Adjust bbox ratio not in the [0, 1] interval."

    print(f"Exporting: {split}")

    X, y, metadata = process_split(root_dir, split, num_max_samples=num_max_samples, debug=debug, adjust_bbox_ratio=adjust_bbox_ratio)
    export_split(X, y, metadata, export_dir, split)


def process_split(root_dir: Path, split: str, num_max_samples: Optional[int], debug: bool = False, adjust_bbox_ratio: float = 0.0):
    """Function used to compute the embeddings of COCO cropped bounding boxes from their afferent images.

    Args:
        split (str): COCO split, choices: ('train', 'val')
        root_dir (Path): root directory of COCO data
        export_dir (Path): root directory where the parsed data will be saved
        num_max_samples (Optional[int], optional): parameter used to limit the maximum number of exports. If None, all the samples will be exported.
        debug (bool, optional): if true, some visualizations will be shown to debug the bounding boxes. Defaults to False.
        adjust_bbox_ratio (float, optional): ratio used to increase the bounding box size to add additional context. Defaults to 0.0.
    """

    annotations_file = get_annotations_path(root_dir, split)

    # Build the COCO utils objects.
    coco=COCO(annotations_file)
    img_ids = coco.getImgIds()
    if bool(num_max_samples):
        np.random.shuffle(img_ids)

    # Build the pretrained model.
    model, preprocess, device = build_resenet50_model()

    # Crop the bounding boxes from all the gives images. 
    X = []
    y = []
    metadata = []
    for img_idx in tqdm(img_ids):
        image_metadata = coco.loadImgs(ids=[img_idx])[0]
        image_path = get_images_path(root_dir, split, image_metadata["file_name"])
        image = Image.open(image_path)
        image = np.array(image)

        if len(image.shape) != 3:
            continue

        ann_ids = coco.getAnnIds(imgIds=[img_idx], iscrowd=False)
        anns = coco.loadAnns(ids=ann_ids)
        for ann in anns:
            category_id = ann["category_id"]
            binary_mask = coco.annToMask(ann)
            ind = np.nonzero(binary_mask)
            
            try:
                x_min = np.min(ind[0])
                x_max = np.max(ind[0])

                y_min = np.min(ind[1])
                y_max = np.max(ind[1])
            except ValueError:  #raised if `ind` is empty.
                continue          
            
            x_min, x_max, y_min, y_max = adjust_bbox(
                bbox=(x_min, x_max, y_min, y_max),
                ratio=adjust_bbox_ratio,
                image_shape=image.shape
            )
            x_min, x_max, y_min, y_max = round(x_min), round(x_max), round(y_min), round(y_max)
            if x_min == x_max or y_min == y_max:
                continue

            crop = image[x_min:x_max, y_min: y_max]

            if debug:
                _, ax = plt.subplots()
                ax.imshow(image)
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.title(f"{category_id=}")
                plt.show()

            crop = np.transpose(crop, (2, 0, 1))

            with torch.no_grad():
                crop = torch.from_numpy(crop).to(device)
                crop = crop.unsqueeze(0)
                crop = preprocess(crop)

                features = model(crop)
                features = features.cpu()
                features = features.detach().numpy()

            X.append(features)
            y.append(category_id)
            metadata.append([img_idx, ann["id"], category_id])

        if bool(num_max_samples) and len(X) >= num_max_samples:
            print(f"Data reached {num_max_samples=}. Early stopping...")
            break 

    # Store the cropped bounding boxes, their category, and metadata.
    X = np.vstack(X)
    y = np.array(y, dtype="int32")
    metadata = pd.DataFrame(metadata, columns=["Image ID", "Annotation ID", "Category ID"])

    return X, y, metadata


def adjust_bbox(bbox: tuple, ratio: float, image_shape: tuple) -> tuple:
    """Utility function that takes as input a bounding box annotation and resizes it with 'ratio.'"""

    assert ratio >= 0.

    if ratio <= 1e-7:
        return bbox

    x_min, x_max, y_min, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    resized_width = width * (1 + ratio)
    resized_height = height * (1 + ratio)

    offset_width = resized_width - width
    offset_height = resized_height - height

    onesided_offset_width = offset_width / 2
    onesided_offset_height = offset_height / 2

    x_min -= onesided_offset_width
    y_min -= onesided_offset_height
    x_max += onesided_offset_width
    y_max += onesided_offset_height

    height, width, _ = image_shape
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, width - 1)
    y_max = min(y_max, height - 1)

    return x_min, x_max, y_min, y_max


def get_annotations_path(root_dir: Path, split: str) -> Path:
    """Utility function that computes the path to an annotation file."""

    annotations_file = split_info[split]["annotations_file"]

    return root_dir / "annotations" / annotations_file


def get_images_path(root_dir: Path, split: str, image_name: str) -> Path:
    """Utility function used to compute the path to a given image."""

    return root_dir / "images" / f"{split}2017" / image_name


def export_split(X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame, export_dir: Path, split: str) -> None:
    """Function used to save the processed bounding boxes to disk.

    Args:
        X (np.ndarray): embeddings of the bounding boxes 
        y (np.ndarray): the categories of the bounding boxes
        metadata (pd.DataFrame): COCO metadata about the bounding boxes
        export_dir (Path): directory where the data will be saved
        split (str): COCO split string
    """

    export_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        get_export_path(export_dir, split, "x"), X
        )
    np.save(
        get_export_path(export_dir, split, "y"), y
        )

    metadata_export_path = get_export_path(export_dir, split, 'metadata')
    if not str(metadata_export_path).endswith(".csv"):
        metadata_export_path = f"{metadata_export_path}.csv" 
    metadata.to_csv(metadata_export_path, index=False)


def get_export_path(export_dir: Path, split: str, file_name: str) -> Path:
    """Utility function used to compute the path for a specific exported file."""

    if split == "val":
        split = "test"

    return export_dir / f"{file_name}_{split}"



def build_resenet50_model():
    """Utility function that builds and loads a Resnet50 model pretrained on COCO."""

    # Get the pretrained Resnet50 backbone.
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    pretrained_resnet50 = model.backbone.body

    # Load the pretrained resnet50 backbone in an end-to-end resnet50 architecture.
    resnet50 = models.resnet50()
    resnet50.load_state_dict(pretrained_resnet50.state_dict(), strict=False)

    # Remove the classification head.
    resnet50.fc = nn.Identity()
    resnet50.eval()

    # If available, move the model to the GPU. Otherwise, to the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet50.to(device)

    # Get the preprocessing pipeline.
    preprocess = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()

    return resnet50, preprocess, device


def export_reduced_dimensions(export_dir: Path, to_n_dimensions: int = 256):
    """Function that loads the full-size embeddings, reduces them to the size of the given dimension using PCA and saves them to disk."""

    print("Reducing embedding dimensions using PCA...")
    
    metadata_train = pd.read_csv(export_dir / "metadata_train.csv")
    metadata_test = pd.read_csv(export_dir / "metadata_test.csv")

    X_train = np.load(export_dir / "x_train.npy")
    X_test = np.load(export_dir / "x_test.npy")
    y_train = np.load(export_dir / "y_train.npy")
    y_test = np.load(export_dir / "y_test.npy")

    X_train, X_test = reduce_dimensions(X_train, X_test, to_n_dimensions)

    export_split(X_train, y_train, metadata_train, export_dir / f"reduced_dimensions={to_n_dimensions}", "train")
    export_split(X_test, y_test, metadata_test, export_dir / f"reduced_dimensions={to_n_dimensions}", "test")



def reduce_dimensions(X_train: np.ndarray, X_test: np.ndarray, to_n_dimensions: int) -> np.ndarray:
    """Function that loads the full-size embeddings and reduces them to the size of the given dimension using PCA."""

    assert to_n_dimensions < X_train.shape[1], f"{to_n_dimensions=} < features length = {X_train.shape[1]}"

    pca = PCA(n_components=to_n_dimensions, random_state=42)
    print("Training PCA. Transforming train data...")
    X_train = pca.fit_transform(X_train)
    print("Transforming test data...")
    X_test = pca.transform(X_test)

    return X_train, X_test



if __name__ == "__main__":
    np.random.seed(42)

    args = parser.parse_args()
    args.root_dir = Path(args.root_dir)
    args.export_dir = Path(args.export_dir)
    args.export_dir.mkdir(parents=True, exist_ok=True)

    export(
        split="train",
        root_dir=args.root_dir,
        export_dir=args.export_dir,
        debug=args.debug,
        adjust_bbox_ratio=args.adjust_bbox_ratio,
        )
    export(
        split="val",
        root_dir=args.root_dir,
        export_dir=args.export_dir,
        debug=args.debug, 
        adjust_bbox_ratio=args.adjust_bbox_ratio,
        )
    export_reduced_dimensions(export_dir=args.export_dir, to_n_dimensions=args.reduce_dimensions_to)


######### Run script
### python feature_extraction_segmentation_coco_resnet50_pytorch.py.py --root-dir COCOdataset2017 --export-dir COCOdataset2017/generated
######### Install requirements:
### conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
### conda install matplotlib==3.6.2, pandas==1.5.1, pycocotools==2.0.6, tqdm==4.64.1, scikit-learn==1.1.3, PIL==9.3.0
