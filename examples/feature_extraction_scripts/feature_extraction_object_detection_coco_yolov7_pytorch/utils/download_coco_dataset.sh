#!/bin/bash

workdir="$1"

help () {
    echo "usage: $(basename $0) <coco_dataset_folder_path>"
}

if [ $# -lt 1 ] ;
then
    help
    exit 1
fi

cd $workdir

mkdir images
cd images
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip

unzip train2017.zip
unzip val2017.zip
unzip test2017.zip

rm train2017.zip
rm val2017.zip
rm test2017.zip

cd ../
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip

unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip

unzip image_info_test2017.zip
rm image_info_test2017.zip
