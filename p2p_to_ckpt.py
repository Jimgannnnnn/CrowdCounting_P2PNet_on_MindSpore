import mindspore
import torch
import pandas as pd
import csv
import numpy as np
from mindspore import load_checkpoint,load_param_into_net,save_checkpoint

from P2PNet_ms.models import build_model
import argparse
from P2PNet_ms.run_test import get_args_parser

from p2p_keys_map import p2p_keys_map as key_dict

#load ms_model
parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
ms_model = build_model(args)

#load torch_model
torch_model=torch.load('./weights/SHTechA.pth',map_location='cpu')
#torch_param_dict=torch_model.state_dict()


# 把双方的key值读进来
torch_keys = pd.read_csv('p2p_pth_0725.csv')
mindspore_keys = pd.read_csv('p2p_ckpt_0725.csv')

# 把'num_batches_tracked'参数踢掉
torch_list = []
for index,value in torch_keys.iterrows():
    name = value[-1]
    if not 'num_batches_tracked' in name:
        torch_list.append(name)
# 制作字典
ms_list = []
for index,value in mindspore_keys.iterrows():
    name = value[-1]
    ms_list.append(name)


'''key_dict = {}
if len(torch_list) == len(ms_list):
    for p in range(len(torch_list)):
        key_dict[ms_list[p]] = torch_list[p]'''

ms_param_list={}

for k,v in ms_model.parameters_dict().items():
    torch_k = key_dict[k]
    torch_v = torch_model['model'][torch_k]
    if not isinstance(torch_v,np.ndarray):
        torch_v = torch_v.cpu().numpy()
    ms_v = mindspore.Parameter(torch_v,name=k)
    #ms_model.parameters_dict()[k] = ms_v    #此处
    ms_param_list[k]=ms_v

load_param_into_net(ms_model,ms_param_list)

a=ms_model.parameters_dict()

save_checkpoint(ms_model,'ms_model_bn.ckpt')


#权重中的tensor在赋值时突变,不不不，它是完全没赋值进去