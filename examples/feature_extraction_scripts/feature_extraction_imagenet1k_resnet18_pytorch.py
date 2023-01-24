"""
Using PyTorch, generate the ImageNet-1K ResNet18 classification dataset.
Store generated x_train, x_test, y_train, y_test numpy array to the local storage.
"""

import numpy as np
from pathlib import Path
from urllib.request import urlretrieve
from tqdm.auto import tqdm
import math
import os.path

import torch
import torch.nn as nn
from torchvision import datasets, transforms

def load_imagenet1k_resnet18_classification_dataset(
    ilsvrc2012_img_train_root: Path, 
    ilsvrc2012_img_val_root: Path, 
    root: Path, 
    force_preprocess: bool = False
):
    """
    loads the ImageNet-1K ResNet18 classification dataset for multinomial logistic regression.

    Parameters
    ----------
    root : Path.
        Indicates the local folder where preprocessed files are to be stored.

    force_preprocess : bool, optional.
        Forces preprocessing the dataset.

    Returns
    -------
    Paths to 6 numpy arrays, in the following order:
        x_train_path: dataset's train feaures,
        y_train_path: dataset's train labels,
        x_test_path: dataset's test feaures,
        y_test_path: dataset's test labels,
        relpath_img_train_path: relative paths of the train images, with respect to `ilsvrc2012_img_train_root`,
        relpath_img_test_path: relative paths of the test images, with reespect to `ilsvrc2012_img_test_root`
    """

    def build_resnet18_model():
    
        def conv3x3(in_planes, out_planes, stride=1):
            """3x3 convolution with padding"""
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)    
        
        class BasicBlock(nn.Module):
            expansion = 1
        
            def __init__(self, inplanes, planes, stride=1, downsample=None):
                super(BasicBlock, self).__init__()
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = nn.BatchNorm2d(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = conv3x3(planes, planes)
                self.bn2 = nn.BatchNorm2d(planes)
                self.downsample = downsample
                self.stride = stride
        
            def forward(self, x):
                residual = x
        
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
        
                out = self.conv2(out)
                out = self.bn2(out)
        
                if self.downsample is not None:
                    residual = self.downsample(x)
        
                out += residual
                out = self.relu(out)
        
                return out    
        
        class Bottleneck(nn.Module):
            expansion = 4
        
            def __init__(self, inplanes, planes, stride=1, downsample=None):
                super(Bottleneck, self).__init__()
                self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
                self.bn3 = nn.BatchNorm2d(planes * 4)
                self.relu = nn.ReLU(inplace=True)
                self.downsample = downsample
                self.stride = stride
        
            def forward(self, x):
                residual = x
        
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
        
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)
        
                out = self.conv3(out)
                out = self.bn3(out)
        
                if self.downsample is not None:
                    residual = self.downsample(x)
        
                out += residual
                out = self.relu(out)
        
                return out    
        
        class ResNet(nn.Module):
            def __init__(self, block, layers, num_classes=1000):
                super(ResNet, self).__init__()
                self.inplanes = 64
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.layer1 = self._make_layer(block, 64, layers[0])
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
                self.avgpool = nn.AvgPool2d(7, stride=1)
                self.fc = nn.Linear(512 * block.expansion, num_classes)
        
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()
        
            def _make_layer(self, block, planes, blocks, stride=1):
                downsample = None
                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                                stride=stride, bias=False),
                        nn.BatchNorm2d(planes * block.expansion),
                    )
        
                layers = []
                layers.append(block(self.inplanes, planes, stride, downsample))
                self.inplanes = planes * block.expansion
                for i in range(1, blocks):
                    layers.append(block(self.inplanes, planes))
        
                return nn.Sequential(*layers)
        
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
        
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
        
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                
                return x
        
        resnet18_model = ResNet(BasicBlock, [2, 2, 2, 2])
        
        weights_folder = root / 'weights'
        weights_folder.mkdir(parents=True, exist_ok=True)
        resnet18_weights_filepath = root / 'weights' / 'resnet18_weights.pth'
        if not resnet18_weights_filepath.exists():
            urlretrieve('https://download.pytorch.org/models/resnet18-5c106cde.pth', resnet18_weights_filepath.as_posix())
        resnet18_model.load_state_dict(torch.load(resnet18_weights_filepath.as_posix()))
        resnet18_model.eval()
        if torch.cuda.is_available():
            resnet18_device = torch.device("cuda")
            resnet18_device_non_blocking = True
        else:
            resnet18_device = torch.device("cpu")
            resnet18_device_non_blocking = False
        resnet18_model.to(resnet18_device)
        
        return resnet18_model, resnet18_device, resnet18_device_non_blocking

    def get_imagenet1k_data_loaders(train_img_folder : Path, val_img_folder : Path):
        normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        imagenet1k_train_dataset = datasets.ImageFolder(
            str(train_img_folder),
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_transform
            ])
        )
        imagenet1k_test_dataset = datasets.ImageFolder(
            str(val_img_folder),
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize_transform
            ])
        )
        
        imagenet1k_train_loader = torch.utils.data.DataLoader(
            imagenet1k_train_dataset,
            batch_size=1024,
            num_workers=1,
        )
        imagenet1k_test_loader = torch.utils.data.DataLoader(
            imagenet1k_test_dataset,
            batch_size=1000,
            num_workers=1,
        )
        return imagenet1k_train_dataset, imagenet1k_test_dataset, imagenet1k_train_loader, imagenet1k_test_loader

    # Define internal folders and create them
    generated_path: Path = root / 'ImageNet_1K' / 'generated'

    # If the above paths don't exist create them
    root.mkdir(parents=True, exist_ok=True)
    generated_path.mkdir(parents=True, exist_ok=True)

    # Activate printing debug info
    print_debug = True

    # Dataset processed files
    all_file_paths = {
        'x_train': generated_path / 'x_train.npy',
        'y_train': generated_path / 'y_train.npy',
        'x_test': generated_path / 'x_test.npy',
        'y_test': generated_path / 'y_test.npy',
        'img_train' : generated_path / 'img_train.npy',
        'img_test' : generated_path / 'img_test.npy',
    }

    all_dataset_files = [all_file_paths[key] for key in all_file_paths]
    if (not all([f.is_file() for f in all_dataset_files])) or force_preprocess:
        if print_debug:
            print('[warning]: dataset feature and label files not found; building them')

        # Build the model
        model, device, non_blocking = build_resnet18_model()

        # Build the ImageNet-1K data loaders
        train_dataset, test_dataset, train_loader, test_loader = \
            get_imagenet1k_data_loaders(ilsvrc2012_img_train_root, ilsvrc2012_img_val_root)
        
        relative_train_images_paths = np.array([os.path.relpath(image_tuple[0], ilsvrc2012_img_train_root) for image_tuple in train_dataset.imgs])
        relative_test_images_paths = np.array([os.path.relpath(image_tuple[0], ilsvrc2012_img_val_root) for image_tuple in test_dataset.imgs])
        np.save(str(all_file_paths['img_train']), relative_train_images_paths)
        np.save(str(all_file_paths['img_test']), relative_test_images_paths)
        del relative_train_images_paths, relative_test_images_paths

        x_train_list, y_train_list = [], []
        x_test_list, y_test_list = [], []

        with torch.no_grad():
            for x_batch, y_batch in tqdm(train_loader):
                x_batch = x_batch.to(device, non_blocking = non_blocking)
                output = model(x_batch)
                x_train_list.append(output.cpu().detach().numpy())
                y_train_list.append(y_batch.detach().numpy().flatten())

            x_train = np.vstack(x_train_list)
            y_train = np.hstack(y_train_list)
            np.save(str(all_file_paths['x_train']), x_train)
            np.save(str(all_file_paths['y_train']), y_train)
            del x_train, y_train

            for x_batch, y_batch in tqdm(test_loader):
                x_batch = x_batch.to(device, non_blocking = non_blocking)
                output = model(x_batch)
                x_test_list.append(output.cpu().detach().numpy())
                y_test_list.append(y_batch.detach().numpy().flatten())

            x_test = np.vstack(x_test_list)
            y_test = np.hstack(y_test_list)
            np.save(str(all_file_paths['x_test']), x_test)
            np.save(str(all_file_paths['y_test']), y_test)
            del x_test, y_test

            if print_debug:
                print(f"[info   ]: {all_file_paths['x_train'].as_posix()} has been created")
                print(f"[info   ]: {all_file_paths['y_train'].as_posix()} has been created")
                print(f"[info   ]: {all_file_paths['x_test'].as_posix()} has been created")
                print(f"[info   ]: {all_file_paths['y_test'].as_posix()} has been created")
                print(f"[info   ]: {all_file_paths['img_train'].as_posix()} has been created")
                print(f"[info   ]: {all_file_paths['img_test'].as_posix()} has been created")                

    return all_file_paths['x_train'], all_file_paths['x_test'], all_file_paths['y_train'], \
           all_file_paths['x_train'], all_file_paths['img_train'], all_file_paths['img_test']


def main():
    # Define local folder for storing dataset files
    data_path = Path("/home/frank/work/projects/for_library_release/imagenet1k_resnet18/data")
    x_train, x_test, y_train, y_test, img_train, img_test = load_imagenet1k_resnet18_classification_dataset(
        ilsvrc2012_img_train_root = Path("/home/frank/work/data/datasets/ImageNet-1K-partial/ILSVRC2012_img_train"),
        ilsvrc2012_img_val_root = Path("/home/frank/work/data/datasets/ImageNet-1K-partial/ILSVRC2012_img_val"), 
        root=data_path)
    print(f'x_train={x_train}\nx_test={x_test}\ny_train={y_train}\ny_test={y_test}\nimg_train={img_train}\nimg_test={img_test}')


if __name__ == "__main__":
    main()
