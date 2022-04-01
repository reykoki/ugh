import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torchinfo import summary

from dataset import VizWizDataset
from model import VizWizNet
from train import train_model
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--embed_dim', type=int, default=500)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--save_loc', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_data_dict = pickle.load(open('pre_proc_train_data0.pickle', 'rb'))
    train_ds = VizWizDataset(train_data_dict)
    val_data_dict = pickle.load(open('pre_proc_val_data0.pickle', 'rb'))
    val_ds = VizWizDataset(val_data_dict)

    model = VizWizNet(train_ds, args.embed_dim, args.num_hid).to(device)
    #model.w_emb.init_embedding('data/glove6b_init_300d.npy')

    train_dataloader = DataLoader(train_ds, args.batch_size, shuffle=True)
    val_dataloader =  DataLoader(val_ds, args.batch_size, shuffle=True)
    train_model(train_dataloader, val_dataloader,
                model, args.n_epochs, args.save_loc, device)

