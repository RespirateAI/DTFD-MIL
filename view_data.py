import pickle
import torch
import numpy as np
import os

from Data import CustomDataset, custom_collate
from torch.utils.data import DataLoader


def reOrganize_mDATA(mDATA, destination_directory):

    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)
        print(f"Slidename: {slide_name}")

        if slide_name.startswith("tumor"):
            label = 1
        elif slide_name.startswith("normal"):
            label = 0
        else:
            raise RuntimeError("Undefined slide type")
        # Label.append(label)

        patch_data_list = mDATA[slide_name]
        print(f"Patch count: {len(patch_data_list)}")
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch["feature"])
            featGroup.append(tfeat.unsqueeze(0))
        featGroup = torch.cat(featGroup, dim=0)  ## numPatch x fs
        # FeatList.append(featGroup)

        print(f"Feature shape: {featGroup.shape}")

        # Save featGroup and Label to a NumPy file
        file_path = os.path.join(destination_directory, f"{slide_name}.npy")
        np.save(file_path, {"featGroup": featGroup.numpy(), "label": label})

        print(f"Saved {slide_name}")

    return SlideNames, FeatList, Label


def reOrganize_mDATA_test(mDATA, destination_directory, testMask_dir):

    tumorSlides = os.listdir(testMask_dir)
    tumorSlides = [sst.split(".")[0] for sst in tumorSlides]

    SlideNames = []
    FeatList = []
    Label = []

    count = 0
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)
        print(f"Slidename: {slide_name}")

        if slide_name in tumorSlides:
            label = 1
            count += 1
            print("Tumor")
        else:
            label = 0
        # Label.append(label)

        patch_data_list = mDATA[slide_name]
        print(f"Patch count: {len(patch_data_list)}")
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch["feature"])
            featGroup.append(tfeat.unsqueeze(0))
        featGroup = torch.cat(featGroup, dim=0)  ## numPatch x fs
        # FeatList.append(featGroup)

        print(f"Feature shape: {featGroup.shape}")

        # Save featGroup and Label to a NumPy file
        file_path = os.path.join(destination_directory, f"{slide_name}.npy")
        np.save(file_path, {"featGroup": featGroup.numpy(), "label": label})

        print(f"Saved {slide_name}")

    print(f"Total tumor slides: {count}")

    return SlideNames, FeatList, Label


if __name__ == "__main__":

    filepath = "/media/ravindu/SSD-PLU3/Lung_CT_Dataset/Lung-PET-CT-Dx-NBIA-Manifest-122220/DTFD-MIL/Data/mDATA_test.pkl"
    destpath = "Data/split_test"

    train_data = pickle.load(open(filepath, "rb"))

    # SlideNames_train, FeatList_train, Label_train = reOrganize_mDATA(
    #     train_data, destpath
    # )

    SlideNames_train, FeatList_train, Label_train = reOrganize_mDATA_test(
        train_data, destpath, "Data/mask"
    )

    print("Length: ", len(SlideNames_train))

    # data = CustomDataset("./Data/split")

    # dataloader = DataLoader(data, 5, collate_fn=custom_collate)

    # for batch in dataloader:
    #     slide_names_batch = batch["slide_name"]
    #     features_batch = batch["features"]
    #     labels_batch = batch["label"]

    #     print(len(slide_names_batch))
    #     print(slide_names_batch)
    #     print(len(features_batch[0]))
    #     features_np = torch.tensor(features_batch[1])
    #     print(features_np.shape)
