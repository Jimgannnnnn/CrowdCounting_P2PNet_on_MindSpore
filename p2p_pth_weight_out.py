import torch
from run_test import get_args_parser
import argparse
import pandas as pd
from models import build_model


parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
args = parser.parse_args()


pytorch_model=build_model(args)
pytorch_model.cuda()

pytorch_weights_dict = pytorch_model.state_dict()
param_torch = pytorch_weights_dict.keys()
param_torch_lst = pd.DataFrame(param_torch)
param_torch_lst.to_csv('p2p_pth_0725.csv')

