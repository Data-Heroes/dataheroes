import os
from pathlib import Path

import numpy as np
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline


def load_ag_news_embeddings_huggingface(root: Path):
    """
    Load and export the AG News text embeddings from HuggingFace.

    :param root: Path to the root folder for saving data
    :return:
        Numpy arrays of dataset's X and y.
    """

    def export(dataset, suffix: str, output_dir: str):
        dataloader = DataLoader(dataset, batch_size=16)

        aggregated_raw_input = []
        aggregated_features = []
        for batch in tqdm(dataloader):
            batch = batch['text']
            aggregated_raw_input.extend(batch)

            results = feature_extractor(batch)

            for result in results:
                result = np.array(result)
                features = np.max(result, axis=1)

                aggregated_features.append(features)

        aggregated_raw_input = np.array(aggregated_raw_input)
        aggregated_features = np.vstack(aggregated_features)
        aggregated_labels = np.array(dataset["label"])

        raw_path = os.path.join(output_dir, f"raw_{suffix}")
        features_path = os.path.join(output_dir, f"x_{suffix}")
        labels_path = os.path.join(output_dir, f"y_{suffix}")

        np.save(raw_path, aggregated_raw_input)
        np.save(features_path, aggregated_features)
        np.save(labels_path, aggregated_labels)

        return aggregated_features, aggregated_labels

    output_dir = os.path.join(root, 'ag_news')
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    dataset = load_dataset(*('ag_news',))

    try:
        train_dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
    except KeyError:
        train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_dataset = train_dataset.shuffle(seed=42).select(
        range(min(len(train_dataset), 100000))
    )
    test_dataset = test_dataset.shuffle(seed=42).select(
        range(min(len(test_dataset), 15000))
    )

    feature_extractor = pipeline(
        task="feature-extraction",
        model="bert-base-cased",
        framework="pt",
        return_tensor=True,
        device='cpu',
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    x_train, y_train = export(train_dataset, suffix="train", output_dir=output_dir)
    x_test, y_test = export(test_dataset, suffix="test", output_dir=output_dir)

    return x_train, x_test, y_train, y_test


def main():
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True, parents=True)
    x_train, x_test, y_train, y_test = load_ag_news_embeddings_huggingface(data_dir)
    print('x_train:{} x_test:{} y_train:{} y_test:{}'.format(
        x_train.shape, x_test.shape, y_train.shape, y_test.shape))


if __name__ == "__main__":
    main()
