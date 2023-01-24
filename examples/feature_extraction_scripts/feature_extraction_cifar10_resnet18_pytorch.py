"""
Using PyTorch, generate the CIFAR10 ResNet18 classification dataset.
Store generated x_train, x_test, y_train, y_test numpy array to the local storage.
"""

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def load_cifar10_res_net18_classifier(root: Path, force_preprocess: bool = False):
    """
    loads the CIFAR10 ResNet18 classification dataset for multinomial logistic regression.

    Parameters
    ----------
    root : Path.
        Indicates the local folder where caching and preprocessing files are to be stored.

    force_preprocess : bool, optional.
        Forces preprocessing the dataset.

    Returns
    -------
    Numpy arrays of dataset's X and y.

    """

    def build_resnet18_model():
        # Load the ResNet18 pre-trained weights
        weights = ResNet18_Weights.DEFAULT
        # Build a ResNet18 model, with the pre-trained weights and set it to be used for inference
        resnet18_model = resnet18(weights=weights)
        resnet18_model.eval()
        # Bypass the final classification layer of the model
        resnet18_model.fc = nn.Identity()
        # Set model to use GPU, if available
        resnet18_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet18_model.to(resnet18_device)

        # Define the preprocessing transformation
        resnet18_preprocess = weights.transforms()

        return resnet18_model, resnet18_preprocess, resnet18_device

    def get_cifar10_data_loaders(cifar10_dataset_path: Path):
        # Load CIFAR10 train set and test set
        train_dataset = datasets.CIFAR10(str(cifar10_dataset_path), train=True, download=True,
                                         transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10(str(cifar10_dataset_path), train=False, download=True,
                                        transform=transforms.ToTensor())
        # Build dataloaders for batches delivery
        cifar10_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
        cifar10_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

        return cifar10_train_loader, cifar10_test_loader

    # Define internal folders and create them
    downloaded_path: Path = root / 'downloaded'
    generated_path: Path = root / 'generated'
    train_images_path: Path = generated_path / 'images' / 'train'
    test_images_path: Path = generated_path / 'images' / 'test'

    # If the above paths don't exist create them
    root.mkdir(parents=True, exist_ok=True)
    downloaded_path.mkdir(parents=True, exist_ok=True)
    generated_path.mkdir(parents=True, exist_ok=True)
    train_images_path.mkdir(parents=True, exist_ok=True)
    test_images_path.mkdir(parents=True, exist_ok=True)

    # Activate printing debug info
    print_debug = True

    # Dataset processed files
    all_file_paths = {
        'x_train': generated_path / 'x_train.npy',
        'y_train': generated_path / 'y_train.npy',
        'x_test': generated_path / 'x_test.npy',
        'y_test': generated_path / 'y_test.npy'
    }

    all_dataset_files = [all_file_paths[key] for key in all_file_paths]
    if (not all([f.is_file() for f in all_dataset_files])) or force_preprocess:
        if print_debug:
            print('[warning]: dataset feature and label files not found; building them')

        # Build the model
        model, preprocess, device = build_resnet18_model()

        # Build CIFAR10 data loaders
        train_loader, test_loader = get_cifar10_data_loaders(cifar10_dataset_path=downloaded_path)

        x_train_list, y_train_list = [], []
        x_test_list, y_test_list = [], []

        with torch.no_grad():
            train_index = 0
            for x_batch, y_batch in tqdm(train_loader):
                input_batch = preprocess(x_batch).to(device)
                output = model(input_batch).cpu()
                x_train_list.append(output.detach().numpy())
                y_train_list.append(np.expand_dims(y_batch.detach().numpy(), axis=1).astype(np.uint8))
                for i in range(x_batch.shape[0]):
                    img = transforms.ToPILImage()(x_batch[i])
                    img.save(train_images_path / f"train_{train_index:06d}-labelid_{y_batch[i].detach()}.png", "PNG")
                    train_index = train_index + 1

            x_train = np.vstack(x_train_list)
            y_train = np.squeeze(np.vstack(y_train_list))

            test_index = 0
            for x_batch, y_batch in tqdm(test_loader):
                input_batch = preprocess(x_batch).to(device)
                output = model(input_batch).cpu()
                x_test_list.append(output.detach().numpy())
                y_test_list.append(np.expand_dims(y_batch.detach().numpy(), axis=1).astype(np.uint8))
                for i in range(x_batch.shape[0]):
                    img = transforms.ToPILImage()(x_batch[i])
                    img.save(test_images_path / f"test_{test_index:06d}-labelid_{y_batch[i].detach()}.png", "PNG")
                    test_index = test_index + 1

            x_test = np.vstack(x_test_list)
            y_test = np.squeeze(np.vstack(y_test_list))

        np.save(str(all_file_paths['x_train']), x_train)
        np.save(str(all_file_paths['y_train']), y_train)
        np.save(str(all_file_paths['x_test']), x_test)
        np.save(str(all_file_paths['y_test']), y_test)

        if print_debug:
            print(f"[info   ]: {all_file_paths['x_train'].as_posix()} has been created")
            print(f"[info   ]: {all_file_paths['y_train'].as_posix()} has been created")
            print(f"[info   ]: {all_file_paths['x_test'].as_posix()} has been created")
            print(f"[info   ]: {all_file_paths['y_test'].as_posix()} has been created")

    x_train: np.ndarray = np.load(str(all_file_paths['x_train']))
    y_train: np.ndarray = np.load(str(all_file_paths['y_train']))
    x_test: np.ndarray = np.load(str(all_file_paths['x_test']))
    y_test: np.ndarray = np.load(str(all_file_paths['y_test']))

    return x_train, x_test, y_train, y_test


def main():
    # Define local folder for storing dataset files
    data_path = Path('../data/cifar10_resnet18')
    x_train, x_test, y_train, y_test = load_cifar10_res_net18_classifier(root=data_path)
    print('x_train:{} x_test:{} y_train:{} y_test:{}'.format(
        x_train.shape, x_test.shape, y_train.shape, y_test.shape))


if __name__ == "__main__":
    main()
