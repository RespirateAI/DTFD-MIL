import numpy as np

import torch
import matplotlib.pyplot as plt
import seaborn as sns


loaded_numpy_data_dict = np.load(
    "./Attention/attention_weights_resnet_softmax.npy", allow_pickle=True
).item()

keys = loaded_numpy_data_dict.keys()

# convert keys to list
keys = list(keys)
# print(keys)

# name = keys[10]
# print(name)


for key in keys:
    if (
        key
        == "Lung_Dx-G0050&11-06-2010-PET03CBMWholebodyFirstHead_Adult-72773&14_patch"
    ):
        print("Found")
        name = key

print(name)
vals = loaded_numpy_data_dict[name]

tier2_weights = vals["tier 2"]
# print(tier2_weights.shape)

tensor = np.zeros(800)
for i in range(800):
    bag_idx = vals[i][1]
    tensor[i] = vals[i][0] * tier2_weights[0][bag_idx]
    tensor[i] = tensor[i] * 1000  # Multiple by 10^3


# print(tensor.shape)
# print(tensor)

# Reshaping the NumPy array
reshaped_attention = tensor.reshape(10, 10, 8)

# Average across the Z-dimension (third dimension)
average_attention = np.mean(reshaped_attention, axis=2)

# 'average_attention' is already a NumPy array, so no need for conversion
heatmap_data = average_attention


# for i in range(reshaped_attention.shape[2]):
#     slice = reshaped_attention[:, :, i]


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
