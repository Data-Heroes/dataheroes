from abc import ABC, abstractmethod
from pathlib import Path
from enum import IntEnum
from collections import defaultdict
from typing import List, Dict
from dataclasses import dataclass
import pickle

from util import download_url, zip_extractall

class DatasetSplit(IntEnum):
    """
    Enumeration class for dataset splits; 
      train and test splits are considered only
    """

    TRAIN_SPLIT = 0
    TEST_SPLIT  = 1

@dataclass
class ImageAnnotation:
    """
    Class encapsulating an image annotation; stores information for an object within a given image

    Parameters
    ----------
    id: int
        annotation id
    bbox: List[float]
        bounding box for the object; represents a list of 4 floating-point values, indicating:
            - top left x position
            - top left y position
            - bounding box width
            - bounding box height
    category_id : int
        the category of the bounded object, corresponding to dataset's encoding
    """
    id : int
    bbox : List[float]      #   - field containing a list of 4 floats indicating the bounding box for the current object within the dataset image
    category_id : int       #   - field indicating the class of the object

class Dataset(ABC):
    """
    Abstract class encapsulating a dataset for feature extraction using YOLOv8
    """

    def __init__(self, download_folder_path : Path) -> None:
        """
        Dataset constructor

        Parameters:
            download_folder_path: Path
                The path to the folder where the dataset is located or, where it will be downloaded
        """
        self.download_folder_path = download_folder_path
        if not download_folder_path.is_dir():
            raise ValueError("Constructor argument is not a folder")
        super().__init__()

        self.img_id_to_filename : List[Dict[int, str]] = [defaultdict(str), defaultdict(str)]
        """
        img_id_to_filename: List[Dict[int, str]]
            a two elements list, containing dictionaries, one for the training part of the dataset and the other for the test part;
            the two dictionaries have keys represented by image ids and values represented by the filename of the respective image
        """
        self.img_id_to_annotations_list : List[Dict[int, List[ImageAnnotation]]] = [defaultdict(list), defaultdict(list)]
        """
        img_id_to_annotations_list : List[Dict[int, List[ImageAnnotation]]]
            a two elements list, containing dictionaries, one for the training part of the dataset and the other for the test part;
            the two dictionaries have keys represented by image ids and values represented by lists of ImageAnnotation objects;
            more precisely, for each image_id, these dictionary stores a list of all annotations for the given image
        """
    
    def get_annotation_metadata_file(self) -> Path:
        # Create (if it doesn't exist) the folder storing dataset's annotation metadata
        metadata_folder = self.download_folder_path / "annotation_metadata"
        metadata_folder.mkdir(parents=True, exist_ok=True)

        # Define the location of the metadata pickle
        return metadata_folder / "annotation_metadata.pickle"

    @abstractmethod
    def has_test_split(self) -> bool:
        """
        Indicates whether the dataset contains a test sub-set, besides the train sub-set
        """
        pass

    @abstractmethod
    def download_dataset(self) -> bool:
        """
        Download dataset's files and, if needed, extracts them; can be called by the 
        load_annotation_metadata() method during the loading of the annotation metadata,
        for example, is some files of the dataset are nit found
        """
        pass

    @abstractmethod
    def load_annotation_metadata(self):
        """
        Loads self.img_id_to_filename and self.img_id_to_annotations_list from disk;
        if the storage support is not found on the disk, the method constructs 
        self.img_id_to_filename and self.img_id_to_annotations_list followed by 
        storing them on disk (pickle format)
        """
        pass

    @abstractmethod
    def get_image_ids(self, split_type : DatasetSplit) -> List[int]:
        """
        Return a list of all image ids in dataset's indicated split;
        used for iterating over all images

        Parameters:
            split_type: DatasetSplit
                specify the dataset's split 
        """
        pass

    @abstractmethod
    def get_image_path(self, split_type : DatasetSplit, image_id : int) -> Path:
        """
        Returns the path of an image for a given dataset split and image id

        Parameters:
            split_type: DatasetSplit
                specify the dataset's split 
            image_id: int
                id of the image
        """
        pass

    @abstractmethod
    def get_annotations_list(self, split_type : DatasetSplit, image_id : int) -> List[ImageAnnotation]:
        """
        For the given dataset split and image id, returns a list of ImageAnnotation 
        encodings for all objects detected in the respective image

        Parameters:
            split_type: DatasetSplit
                specify the dataset's split 
            image_id: int
                id of the image
        """
        pass

class COCODataset(Dataset):
    def __init__(self, download_folder_path: Path) -> None:
        super().__init__(download_folder_path)
        
        # Load dataset metadata
        self.load_annotation_metadata()

    def has_test_split(self) -> bool:
        return True

    def download_dataset(self):
        """
        Download COCO dataset; the COCODataset class expects the dataset to be in the following structure
        <self.download_folder_path>
        ├── annotations
        │   ├── captions_train2017.json
        │   ├── captions_val2017.json
        │   ├── image_info_test-dev2017.json
        │   ├── image_info_test2017.json
        │   ├── instances_train2017.json
        │   ├── instances_val2017.json
        │   ├── person_keypoints_train2017.json
        │   └── person_keypoints_val2017.json
        └── images
            ├── test2017
            ├── train2017
            └── val2017

        The <annotations> folder contains the extracted content of the archive below:
          http://images.cocodataset.org/annotations/annotations_trainval2017.zip
          http://images.cocodataset.org/annotations/image_info_test2017.zip
        The images folder contains the extracted content of the archives below:
          http://images.cocodataset.org/zips/train2017.zip
          http://images.cocodataset.org/zips/val2017.zip
          http://images.cocodataset.org/zips/test2017.zip
        The COCODataset class will create an additional subfolder of <self.download_folder_path>,
          <annotation_metadata>, where the annotations metadata pickle file is stored
        """

        images_dir = self.download_folder_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for filename in ["train2017.zip", "val2017.zip", "test2017.zip"]:
            url = f"http://images.cocodataset.org/zips/{filename}"
            file_path = images_dir / f"{filename}"
            file_path = download_url(url, file_path)
            zip_extractall(file_path, images_dir, delete_zipfile=True)
        
        for filename in ["annotations_trainval2017.zip", "image_info_test2017.zip"]:
            url = f"http://images.cocodataset.org/annotations/{filename}"
            file_path = self.download_folder_path / f"{filename}"
            file_path = download_url(url, file_path)
            zip_extractall(file_path, self.download_folder_path, delete_zipfile=True)
    
    def load_annotation_metadata(self):
        """
        Use pycocotools package for preparing the annotation metadata
        """
        from pycocotools.coco import COCO

        def build_split_metadata(split_type : DatasetSplit):
            annotation_file = self.download_folder_path / "annotations" / ("instances_train2017.json" if split_type is DatasetSplit.TRAIN_SPLIT else "instances_val2017.json")
            if split_type == DatasetSplit.TRAIN_SPLIT:
                # When constructing the pycocotool's COCO object for the train sub-set,
                #   if the annotation file is not found in it's expected location, the 
                #   entire dataset is downloaded first
                if not annotation_file.is_file():
                    self.download_dataset()
            if not annotation_file.is_file():
                raise ValueError(f"Annotation file {annotation_file} not found.")
            
            # Build the COCO dataset object, using the provided annotation JSON file
            coco_obj = COCO(annotation_file)

            # Iterate over all image ids and populate the self.img_id_to_filename[split_type] dict
            for img_id in coco_obj.getImgIds():
                self.img_id_to_filename[split_type][img_id] = coco_obj.loadImgs(ids=[img_id])[0]["file_name"]
            
            # Iterate over all annotations; only preserve annotations for which label iscrowd == False
            # populate the self.img_id_to_annotations_list[split_type] dict
            for ann_id in coco_obj.getAnnIds(iscrowd=False):
                for ann in coco_obj.loadAnns(ids=ann_id):
                    self.img_id_to_annotations_list[split_type][ann["image_id"]].append(ImageAnnotation(ann["id"], ann["bbox"], ann["category_id"]))

        # Get dataset annotation metadata filepath
        annotation_metadata_pickle_filepath = self.get_annotation_metadata_file()

        # If the annotation metadata file is not found, build the data and save it to disk
        if not annotation_metadata_pickle_filepath.is_file():
            build_split_metadata(DatasetSplit.TRAIN_SPLIT)
            if self.has_test_split():
                build_split_metadata(DatasetSplit.TEST_SPLIT)
            
            with open(annotation_metadata_pickle_filepath, 'wb') as fp:
                pickle.dump(self.img_id_to_filename, fp)
                pickle.dump(self.img_id_to_annotations_list, fp)
        else:
            with open(annotation_metadata_pickle_filepath, 'rb') as fp:
                self.img_id_to_filename = pickle.load(fp)
                self.img_id_to_annotations_list = pickle.load(fp)
    
    def get_image_ids(self, split_type : DatasetSplit) -> list:
        return self.img_id_to_filename[split_type].keys()

    def get_image_path(self, split_type : DatasetSplit, image_id : int) -> Path:
        if split_type is DatasetSplit.TRAIN_SPLIT:
            return self.download_folder_path / "images" / "train2017" / self.img_id_to_filename[split_type][image_id]
        else:
            return self.download_folder_path / "images" / "val2017" / self.img_id_to_filename[split_type][image_id]
    
    def get_annotations_list(self, split_type : DatasetSplit, image_id : int) -> List[ImageAnnotation]:
        return self.img_id_to_annotations_list[split_type][image_id]
        

if __name__ == "__main__":
    root_dir = Path("../data/coco_dataset")
    root_dir.mkdir(parents=True, exist_ok=True)

    coco_dataset = COCODataset(root_dir)
    