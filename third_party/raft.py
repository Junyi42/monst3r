
import sys
import argparse
import torch
import json
from os.path import dirname, join
RAFT_PATH_ROOT = join(dirname(__file__), 'RAFT')
RAFT_PATH_CORE = join(RAFT_PATH_ROOT, 'core')
sys.path.append(RAFT_PATH_CORE)
from raft import RAFT, RAFT2  # nopep8
from utils.utils import InputPadder  # nopep8

# %%
# utility functions

def json_to_args(json_path):
    # return a argparse.Namespace object
    with open(json_path, 'r') as f:
        data = json.load(f)
    args = argparse.Namespace()
    args_dict = args.__dict__
    for key, value in data.items():
        args_dict[key] = value
    return args

def parse_args(parser):
    entry = parser.parse_args(args=[])
    json_path = entry.cfg
    args = json_to_args(json_path)
    args_dict = args.__dict__
    for index, (key, value) in enumerate(vars(entry).items()):
        args_dict[key] = value
    return args

def get_input_padder(shape):
    return InputPadder(shape, mode='sintel')


def load_RAFT(model_path=None):
    if model_path is None or 'M' not in model_path: # RAFT1
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint", default=model_path)
        parser.add_argument('--path', help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision',
                            action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true',
                            help='use efficient correlation implementation')
        
        # Set default value for --model if model_path is provided
        args = parser.parse_args(
            ['--model', model_path if model_path else join(RAFT_PATH_ROOT, 'models', 'raft-sintel.pth'), '--path', './'])
        
        net = RAFT(args)
    else: # RAFT2
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', help='experiment configure file name', default="third_party/RAFT/core/configs/congif_spring_M.json")
        parser.add_argument('--model', help='checkpoint path', default=model_path)
        parser.add_argument('--device', help='inference device', type=str, default='cpu')
        args = parse_args(parser)
        net = RAFT2(args)

    if torch.cuda.is_available():
        state_dict = torch.load(args.model)
    else:
        state_dict = torch.load(args.model, map_location="cpu")
    print('Loaded pretrained RAFT model from', args.model)
    new_state_dict = {}
    for k in state_dict:
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = state_dict[k]
    net.load_state_dict(new_state_dict)
    return net.eval()

if __name__ == "__main__":
    net = load_RAFT(model_path='third_party/RAFT/models/Tartan-C-T432x960-M.pth')
    print(net)