import os
import numpy as np
from torch.utils.data import Dataset


def custom_collate(batch):
    slide_names = [item["slide_name"] for item in batch]
    labels = [item["label"] for item in batch]
    features = [item["features"] for item in batch]

    return {"slide_name": slide_names, "features": features, "label": labels}


class CustomDataset(Dataset):
    def __init__(self, destination_directory):
        self.destination_directory = destination_directory
        self.file_list = [
            filename
            for filename in os.listdir(destination_directory)
            if filename.endswith(".npy")
        ]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.destination_directory, file_name)

        data = np.load(file_path, allow_pickle=True).item()
        slide_name = file_name[:-4]  # Slide name taken from file name
        features = data[
            "featGroup"
        ].tolist()  # Convert to list because Variable size tensors cannot be concatenated
        label = data["label"]

        return {"slide_name": slide_name, "features": features, "label": label}
