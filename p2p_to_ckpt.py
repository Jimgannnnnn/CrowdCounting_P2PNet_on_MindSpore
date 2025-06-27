import mindspore
import torch
import pandas as pd
import csv
import numpy as np
from mindspore import load_checkpoint,load_param_into_net,save_checkpoint

from P2PNet_ms.models import build_model          #从P2PNet_ms文件夹中调用模型加载的接口，这里我没有上传
import argparse
from P2PNet_ms.run_test import get_args_parser    #用于导入模型加载时的一些参数

from p2p_keys_map import p2p_keys_map as key_dict

#load ms_model
parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])#参数
args = parser.parse_args()
ms_model = build_model(args)    #带参数加载mindspore版本模型

#load torch_model weight
torch_model=torch.load('./weights/SHTechA.pth',map_location='cpu')    #torch版本的模型并不需要加载，我们只需要读权重即可





ms_param_list={}
#按照映射迁移权重数值到一个字典里
for k,v in ms_model.parameters_dict().items():
    torch_k = key_dict[k]
    torch_v = torch_model['model'][torch_k]
    if not isinstance(torch_v,np.ndarray):
        torch_v = torch_v.cpu().numpy()
    ms_v = mindspore.Parameter(torch_v,name=k)
    ms_param_list[k]=ms_v

#将迁移好的“索引-数据”字典（类型我记得需要是Ordered Dict？不太记得了）导入mindspore模型
load_param_into_net(ms_model,ms_param_list)

#保存为ckpt文件
save_checkpoint(ms_model,'ms_model_bn.ckpt')


