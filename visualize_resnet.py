import numpy as np

import torch
import matplotlib.pyplot as plt
import seaborn as sns


loaded_numpy_data_dict = np.load(
    "./Attention/attention_weights_resnet_1.npy", allow_pickle=True
).item()

# Convert NumPy arrays to PyTorch tensors
loaded_data_dict = {
    key: torch.from_numpy(value) for key, value in loaded_numpy_data_dict.items()
}

keys = loaded_data_dict.keys()

# convert keys to list
keys = list(keys)
# print(keys)

# name = keys[10]
# print(name)


for key in keys:
    if key == "Lung_Dx-A0176&06-28-2009-PET03WholebodyFirstHead_Adult-19609&8_patch":
        print("Found")
        name = key

print(name)
tensor = loaded_data_dict[name]

# Reshape the tensor into a 3D tensor of size (13, 13, 9)
reshaped_attention = tensor.resize(10, 10, 8)

# Average across the Z-dimension (third dimension)
average_attention = torch.mean(reshaped_attention, dim=2)


# for i in range(reshaped_attention.shape[2]):
#     slice = reshaped_attention[:, :, i]

# Convert the PyTorch tensor to a NumPy array for heatmap visualization
heatmap_data = average_attention.numpy()

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    heatmap_data,
    cmap="viridis",
    annot=True,
    fmt=".2f",
    cbar_kws={"label": "Average Attention"},
)
plt.title("Average Attention Across Patches")
plt.savefig("attention_heatmap_resnet_6.png")

plt.show()

# Save this as a png
