import os

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SwinTransformerSys
from utils import parse_args, RainDataset

'''
- This is the demo file for testing the model on a single image.
- Please see the utils.py for passing the arguments.
- Please download the model file from the link given in the ReadMe file.
'''

def run_inference(model, data_loader):
    model.eval()
    with torch.no_grad():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        for rain, _, _, h, w in test_bar:        
            rain = rain.cuda()
            out = torch.clamp((torch.clamp(model(rain)[:, :, :h, :w], 0, 1).mul(255)), 0, 255).byte()
            Image.fromarray(out.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()).save('result/demo-result.png')


if __name__ == '__main__':
    args = parse_args()
    test_dataset = RainDataset(args.data_path, args.data_name, 'test', args.patch_size[0])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    model = SwinTransformerSys().cuda()

    if args.model_file:
        model.load_state_dict(torch.load(args.model_file))
        run_inference(model, test_loader)
    else:
        raise ValueError('Please input the model file!')