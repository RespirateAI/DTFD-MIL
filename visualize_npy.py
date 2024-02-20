import numpy as np


filepath = "Attention/attention_weights_resnet_1.npy"


data = np.load(filepath, allow_pickle=True)


print(data.shape)

print(data)
