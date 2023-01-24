"""
Using TensorFlow and Keras, generate the CIFAR10 ResNet18 classification dataset.
Store generated x_train, x_test, y_train, y_test numpy array to the local storage.

"""

import numpy as np
from pathlib import Path
import tensorflow as tf
from classification_models.keras import Classifiers
from PIL import Image
from tqdm.auto import tqdm


def load_cifar10_res_net18_classifier(root: Path, force_preprocess: bool = False):
    """
    loads the CIFAR10 ResNet18 classification dataset for multinomial logistic regression.

    Parameters
    ----------
    root : Path.
        Indicates the local folder where caching and preprocessing files are to be stored

    force_preprocess : bool, optional.
        Forces preprocessing the dataset.

    Returns
    -------
    Numpy arrays of dataset's X and y.
    """

    def build_res_net18_model():
        # Load the ResNet18 pre-trained model
        resnet18, preprocess_input = Classifiers.get('resnet18')
        base_model = resnet18((224, 224, 3), weights='imagenet')
        # Bypass the final classification layer of the model
        resnet18_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('pool1').output)

        # Define the preprocessing transformation
        def resnet18_preprocess(data):
            return preprocess_input(tf.image.resize(data, [224, 224]).numpy())

        return resnet18_model, resnet18_preprocess

    def get_cifar10_data_loaders():
        # Load CIFAR10 train set and test set
        (x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) = tf.keras.datasets.cifar10.load_data()
        train_dataset = tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensor_slices(x_train_cifar10), tf.data.Dataset.from_tensor_slices(y_train_cifar10)))
        test_dataset = tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensor_slices(x_test_cifar10), tf.data.Dataset.from_tensor_slices(y_test_cifar10)))
        # Build dataloaders for batches delivery
        cifar10_train_loader = train_dataset.batch(64)
        cifar10_test_loader = test_dataset.batch(64)

        return cifar10_train_loader, cifar10_test_loader

    # Define internal folders and create them
    generated_path: Path = root / 'generated'
    train_images_path: Path = generated_path / 'images' / 'train'
    test_images_path: Path = generated_path / 'images' / 'test'

    # If the above paths don't exist create them
    root.mkdir(parents=True, exist_ok=True)
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
        model, preprocess = build_res_net18_model()

        # Build CIFAR10 data loaders
        train_loader, test_loader = get_cifar10_data_loaders()

        x_train_list, y_train_list = [], []
        x_test_list, y_test_list = [], []

        train_index = 0
        for x_batch, y_batch in tqdm(train_loader):
            input_batch = preprocess(x_batch)
            output = model.predict_on_batch(input_batch)
            x_train_list.append(output)
            y_train_list.append(y_batch.numpy())
            for i in range(x_batch.shape[0]):
                img = Image.fromarray(x_batch[i].numpy())
                img.save(train_images_path / f"train_{train_index:06d}-labelid_{y_batch[i][0]}.png", "PNG")
                train_index = train_index + 1
        x_train = np.vstack(x_train_list)
        y_train = np.squeeze(np.vstack(y_train_list))

        test_index = 0
        for x_batch, y_batch in tqdm(test_loader):
            input_batch = preprocess(x_batch)
            output = model.predict_on_batch(input_batch)
            x_test_list.append(output)
            y_test_list.append(y_batch.numpy())
            for i in range(x_batch.shape[0]):
                img = Image.fromarray(x_batch[i].numpy())
                img.save(test_images_path / f"test_{test_index:06d}-labelid_{y_batch[i][0]}.png", "PNG")
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
