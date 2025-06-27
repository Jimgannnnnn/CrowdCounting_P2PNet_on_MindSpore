import mindspore as ms
from P2PNet_ms.models import build_model
from P2PNet_ms.run_test import get_args_parser
import argparse
import pandas as pd

parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

mindspore_model=build_model(args)
prams_ms = mindspore_model.parameters_dict().keys()
prams_ms_lst = pd.DataFrame(prams_ms)
prams_ms_lst.to_csv('p2p_ckpt_0725.csv')