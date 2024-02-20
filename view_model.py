import torch

torch.multiprocessing.set_sharing_strategy("file_system")
import argparse
import os

import random
from Model.Attention import Attention_Gated as Attention
from Model.Attention import Attention_with_Classifier
from Model.network import Classifier_1fc, DimReduction
import numpy as np

from torchsummary import summary

parser = argparse.ArgumentParser(description="abc")
testMask_dir = ""  ## Point to the Camelyon test set mask location

parser.add_argument("--name", default="abc", type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--isPar", default=False, type=bool)
parser.add_argument("--log_dir", default="./debug_log", type=str)  ## log file path
parser.add_argument("--train_show_freq", default=40, type=int)
parser.add_argument("--droprate", default="0", type=float)
parser.add_argument("--droprate_2", default="0", type=float)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--batch_size_v", default=1, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--num_cls", default=2, type=int)
parser.add_argument("--numGroup", default=4, type=int)
parser.add_argument("--total_instance", default=4, type=int)
parser.add_argument("--numGroup_test", default=4, type=int)
parser.add_argument("--total_instance_test", default=4, type=int)
parser.add_argument("--mDim", default=128, type=int)
parser.add_argument("--grad_clipping", default=5, type=float)
parser.add_argument("--isSaveModel", action="store_false")
parser.add_argument("--debug_DATA_dir", default="", type=str)
parser.add_argument("--numLayer_Res", default=0, type=int)
parser.add_argument("--temperature", default=1, type=float)
parser.add_argument("--num_MeanInference", default=1, type=int)
parser.add_argument("--distill_type", default="AFS", type=str)  ## MaxMinS, MaxS, AFS

torch.manual_seed(32)
torch.cuda.manual_seed(32)
np.random.seed(32)
random.seed(32)


def main():
    params = parser.parse_args()

    classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(
        params.device
    )
    attention = Attention(params.mDim).to(params.device)
    attCls = Attention_with_Classifier(
        L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2
    ).to(params.device)

    # Load the saved state dictionaries
    tsave_dict = torch.load(
        os.path.join(params.log_dir, "best_model_lung_norandom_binary.pth")
    )

    # Load the state dictionaries into the respective model components
    classifier.load_state_dict(tsave_dict["classifier"])
    attention.load_state_dict(tsave_dict["attention"])
    attCls.load_state_dict(tsave_dict["att_classifier"])

    # Assuming classifier, attention, and attCls are instances of your models
    # print("Classifier Model:")
    # summary(
    #     classifier, (input_size1, input_size2)
    # )  # Replace input_size1, input_size2 with actual input sizes

    print("\nAttention Model:")
    summary(attention, (169, params.mDim))  # Replace input_size with actual input size

    # print("\nAttention with Classifier Model:")
    # summary(attCls, (input_size,))  # Replace input_size with actual input


if __name__ == "__main__":
    main()
