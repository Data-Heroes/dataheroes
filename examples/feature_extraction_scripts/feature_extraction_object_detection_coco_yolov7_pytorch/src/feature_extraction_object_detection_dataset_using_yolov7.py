import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import pandas as pd
import pickle

from tqdm import tqdm
from typing import Optional
from sklearn.decomposition import IncrementalPCA

from PIL import Image
from pathlib import Path

from models.yolo import Model

from models.experimental import attempt_load
from torch import Tensor
from typing import Any
import cv2

from util import download_url
from datasets import Dataset, COCODataset, DatasetSplit, ImageAnnotation

# CONSTRUCT LOGGER STRUCTURES #################################################
import logging
from time import strftime

def prepareLogging(logFolder : Path, print_debug = True):
    
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=(logging.DEBUG if print_debug else logging.INFO),
                        format='%(asctime)s.%(msecs)03d || %(levelname)-8s || %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=logFolder / f'logger-{Path(__file__).stem}-{strftime("%Y_%m_%d-%H:%M:%S")}.txt',
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if print_debug else logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d || %(levelname)-8s || %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

###############################################################################

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
parser.add_argument(
    "--model_param_location",
    default=None,
    help="Absolute or relative path to the YOLOv7 model.",
)
parser.add_argument(
    "--max_samples_per_extracted_split",
    default=10_000,
    help="Indicates the number of samples an extracted feature or target split will store.",
)


def build_yolo_model(model_param_location: Path):
    """Builds a YOLOv7 model pretrained on COCO, and transforms it for embedded features extraction.
    
    Args:
        model_param_location (Path): path to the YOLOv7 model
    """

    class MyYOLOv7IdentityLayer(torch.nn.Module):
        def __init__(self, final_detect_layer, overwrite_f = None, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            self.i = final_detect_layer.i
            if overwrite_f is None:
                self.f = final_detect_layer.f
            else:
                self.f = overwrite_f

        def forward(self, input: Tensor) -> Tensor:
            return input

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = attempt_load(model_param_location, map_location=device)  # load FP32 model
    # Replace all layers past layer no. 24 with Identities (also replace the m.f link for each layer
    # signifying the source for layer's input)
    current_model_index = 0
    while True:
        current_model_index -= 1
        layer = model.model[current_model_index]
        if layer.i == 24:
            break
        else:
            model.model[current_model_index] = MyYOLOv7IdentityLayer(model.model[current_model_index], overwrite_f=-1)
    model.eval()
    model.to(device)

    return model, device


def extract_features_from_dataset(
        dataset : Dataset,
        model : Model,
        device : torch.device,
        export_dir: Path, 
        max_samples_per_extracted_split:int,
        num_max_samples: Optional[int] = None, 
        debug: bool = False, 
        adjust_bbox_ratio: float = 0.0, 
        imgsz:int = 64, 
        ):
    """Function used to compute the embeddings of COCO cropped bounding boxes from their afferent images and save them to disk.

    Args:
        dataset (Dataset): the dataset object providing access to dataset's images and annotations
        model (Model): the YOLOv7 model
        device (torch.device): the device onto which the model is to be executed
        export_dir (Path): folder path where results of feature extractions are stored
        num_max_samples (Optional[int], optional): parameter used to limit the maximum number of exports. If None, all the samples will be exported.
        debug (bool, optional): if true, some visualizations will be shown to debug the bounding boxes. Defaults to False.
        adjust_bbox_ratio (float, optional): ratio used to increase the bounding box size to add additional context. Defaults to 0.0.
		imgsz (int, optional): side of the square shape the bounding box is resized to. Defaults to 64.
    """

    # Verify the value of the ratio used for increasing the bounding box
    assert 0. <= adjust_bbox_ratio <= 1., "Adjust bbox ratio not in the [0, 1] interval."

    def preprocess_and_export(split_type : DatasetSplit):

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

        logging.info(f"Extracting features from the {('train' if split_type is DatasetSplit.TRAIN_SPLIT else 'test')} split ...")

        feature_dir = export_dir / ("train" if split_type is DatasetSplit.TRAIN_SPLIT else "test") / f"features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        target_dir = export_dir / ("train" if split_type is DatasetSplit.TRAIN_SPLIT else "test") / f"targets"
        target_dir.mkdir(parents=True, exist_ok=True)

        current_split = -1
        current_list_of_features = []
        current_list_of_targets = []
        num_available_slots = max_samples_per_extracted_split
        current_detection = 0
        
        image_ids = dataset.get_image_ids(split_type=split_type)
        if bool(num_max_samples):
            np.random.shuffle(image_ids)

        # Initialize metadata
        metadata = []

        # Build tqdm progress bar
        images_loop = tqdm(image_ids, desc="Image progress")

        for image_id in images_loop:
            image_path = dataset.get_image_path(split_type=split_type, image_id=image_id)
            
            # Read image
            image = Image.open(image_path)
            image = np.array(image)
            # If monochrome image, skip processing
            if len(image.shape) != 3:
                continue
            
            # Get all annotations for given image
            annotations = dataset.get_annotations_list(split_type=split_type, image_id=image_id)

            for annotation in annotations:
                category_id = annotation.category_id
                bbox = annotation.bbox
                original_bbox = (*bbox, )
                x_min, y_min, width, height = bbox
                x_max, y_max = x_min + width, y_min + height
                
                x_min, x_max, y_min, y_max = adjust_bbox(
                    bbox=(x_min, x_max, y_min, y_max),
                    ratio=adjust_bbox_ratio,
                    image_shape=image.shape
                )
                x_min, x_max, y_min, y_max = round(x_min), round(x_max), round(y_min), round(y_max)
                if x_min == x_max or y_min == y_max:
                    continue

                crop = image[y_min:y_max, x_min:x_max, :]
                if debug:
                    _, ax = plt.subplots()
                    ax.imshow(image)
                    rect = patches.Rectangle((original_bbox[0], original_bbox[1]), original_bbox[2], original_bbox[3], linewidth=1, edgecolor='b', facecolor='none')
                    ax.add_patch(rect)
                    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    plt.title(f"{category_id=}")
                    plt.show()

                img = cv2.resize(crop, [imgsz, imgsz])
                img = img.transpose(2, 0, 1)
                img = np.ascontiguousarray(img)

                img = torch.from_numpy(img).to(device)
                img = img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                with torch.no_grad():
                    pred = model(img).cpu().detach().numpy()
                    pred = pred.reshape((-1))

                if num_available_slots == 0:
                    # Create a new extracted features / targets split
                    current_split += 1
                    np.save(feature_dir / f"feature-split_{current_split:03d}", np.array(current_list_of_features, dtype=np.float32))
                    np.save(target_dir / f"target-split_{current_split:03d}", np.array(current_list_of_targets, dtype=np.int32))
                    current_list_of_features = []
                    current_list_of_targets = []
                    num_available_slots = max_samples_per_extracted_split
                num_available_slots -= 1
                current_detection += 1
                
                current_list_of_features.append(pred)
                current_list_of_targets.append(category_id)
                metadata.append([image_id, annotation.id, category_id])
            
            if bool(num_max_samples) and current_detection >= num_max_samples:
                logging.info(f"Data reached {num_max_samples=}. Early stopping...")
                break 
        
        if len(current_list_of_features) != 0:
            # Create the last extracted features / targets split
            current_split += 1
            np.save(feature_dir / f"feature-split_{current_split:03d}", np.array(current_list_of_features, dtype=np.float32))
            np.save(target_dir / f"target-split_{current_split:03d}", np.array(current_list_of_targets, dtype=np.int32))
        
        logging.info(f"In the {('train' if split_type is DatasetSplit.TRAIN_SPLIT else 'test')} split, there were {current_detection} objects for which features were extracted.")

        # Store the metadata.
        metadata = pd.DataFrame(metadata, columns=["Image ID", "Annotation ID", "Category ID"])
        metadata_export_path = export_dir / ("train" if split_type is DatasetSplit.TRAIN_SPLIT else "test") / f"metadata_{('train' if split_type is DatasetSplit.TRAIN_SPLIT else 'test')}.csv"
        metadata.to_csv(metadata_export_path, index=False)

    preprocess_and_export(split_type=DatasetSplit.TRAIN_SPLIT)
    
    if dataset.has_test_split():
        preprocess_and_export(split_type=DatasetSplit.TEST_SPLIT)


def shrink_extracted_features(
        dataset : Dataset,
        export_dir: Path, 
        to_n_dimensions: int = 256
    ):
    """Function that loads the full-size embeddings, reduces them to the size of the given dimension using PCA and saves them to disk.
    
    Args:
        dataset (Dataset): the dataset object providing access to dataset's images and annotations
        export_dir (Path): folder path where results of feature extractions are stored
        to_n_dimensions (int): the number of dimensions to retain in the reduced extracted features
    """
    
    def build_pca_estimator(train_feature_dir : Path, num_train_splits : int):
        pickled_ipca_filepath = export_dir / "pickles" / "ipca.pickle"
        if pickled_ipca_filepath.is_file():
            # Load a pickled IncrementalPCA
            ipca = pickle.load(open(pickled_ipca_filepath,"rb"))
        else:
            # Train an IncremenetalPCA
            ipca = IncrementalPCA(n_components=to_n_dimensions)
            for split in tqdm(range(num_train_splits), "Split progress"):
                x_array = np.load(train_feature_dir / f"feature-split_{split:03d}.npy")
                ipca.partial_fit(x_array)
                x_array = None
            pickle.dump(ipca, open(pickled_ipca_filepath,"wb"))
        
        return ipca

    def reduce_dimension_and_export(split_type : DatasetSplit, feature_dir : Path, target_dir : Path, num_splits : int, rdim_export_dir : Path):
        logging.info(f"Reduce dimension of features extracted from the {('train' if split_type is DatasetSplit.TRAIN_SPLIT else 'test')} split ...")

        list_x = []
        list_y = []
        for split in tqdm(range(num_splits), "Dimension reduction progress"):
            x_array = np.load(feature_dir / f"feature-split_{split:03d}.npy")
            y_array = np.load(target_dir / f"target-split_{split:03d}.npy")
            list_x.append(ipca.transform(x_array))
            list_y.append(y_array)
            x_array = None
            y_array = None
        np.save(rdim_export_dir / f"x_{('train' if split_type is DatasetSplit.TRAIN_SPLIT else 'test')}.npy", np.vstack(list_x))
        list_x = None
        np.save(rdim_export_dir / f"y_{('train' if split_type is DatasetSplit.TRAIN_SPLIT else 'test')}.npy", np.concatenate(list_y))
        list_y = None
    
    # Build paths to relevant folders
    train_feature_dir = export_dir / "train" / f"features"
    train_target_dir = export_dir / "train" / f"targets"
    test_feature_dir = export_dir / "test" / f"features"
    test_target_dir = export_dir / "test" / f"targets"

    # Build folder to store reduced dimension extracted features
    rdim_export_dir = export_dir / f"reduced_dimensions={to_n_dimensions}"
    rdim_export_dir.mkdir(parents=True, exist_ok=True)

    # Determine the number of splits in extracted features of the training split
    num_train_splits = len(list(train_feature_dir.glob('*.npy')))
    if num_train_splits < 1:
        raise ValueError("No split files found for extracted features/targets of the target sub-dataset")
    
    ipca = build_pca_estimator(train_feature_dir, num_train_splits)

    reduce_dimension_and_export(
        split_type=DatasetSplit.TRAIN_SPLIT, 
        feature_dir=train_feature_dir, 
        target_dir=train_target_dir, 
        num_splits=num_train_splits, 
        rdim_export_dir=rdim_export_dir
    )

    if dataset.has_test_split():
        # Determine the number of splits in extracted features of the test split
        num_test_splits = len(list(test_feature_dir.glob('*.npy')))
        reduce_dimension_and_export(
            split_type=DatasetSplit.TEST_SPLIT, 
            feature_dir=test_feature_dir, 
            target_dir=test_target_dir, 
            num_splits=num_test_splits, 
            rdim_export_dir=rdim_export_dir
        )

    metadata_train = pd.read_csv(export_dir / "train" / "metadata_train.csv")
    metadata_train.to_csv(rdim_export_dir  / "metadata_train.csv", index=False)

    metadata_test = pd.read_csv(export_dir / "test" / "metadata_test.csv")
    metadata_test.to_csv(rdim_export_dir  / "metadata_test.csv", index=False)

def main():
    np.random.seed(42)

    args = parser.parse_args()
    
    args.root_dir = Path(args.root_dir)
    args.root_dir.mkdir(parents=True, exist_ok=True)

    args.export_dir = Path(args.export_dir)
    args.export_dir.mkdir(parents=True, exist_ok=True)

    # Create the logging folder
    logs_path = args.export_dir / 'logs'
    logs_path.mkdir(parents=True, exist_ok=True)
    # Start the logger
    prepareLogging(logs_path, print_debug=False)
    
    if args.model_param_location is None:
        args.model_param_location = Path('yolov7.pt')
        if not args.model_param_location.is_file():
            download_url("https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt", args.model_param_location.as_posix(), "Downloading yolov7.pt")
    else:
        args.model_param_location = Path(args.model_param_location)

    # Build the pretrained model
    yolov7_model, yolov7_device = build_yolo_model(model_param_location=args.model_param_location)

    # Build the Dataset object
    coco_dataset=COCODataset(download_folder_path=args.root_dir)

    extract_features_from_dataset(
        dataset=coco_dataset,
        model = yolov7_model,
        device = yolov7_device,
        export_dir=args.export_dir,
        max_samples_per_extracted_split=args.max_samples_per_extracted_split,
        debug=args.debug,
        adjust_bbox_ratio=args.adjust_bbox_ratio,
    )
    shrink_extracted_features(
        dataset=coco_dataset,
        export_dir=args.export_dir, 
        to_n_dimensions=args.reduce_dimensions_to
    )

if __name__ == "__main__":
    main()
