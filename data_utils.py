
from transformers import BertTokenizerFast

import torch
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm

import copy

import json
import numpy as np
import random
import matplotlib.pyplot as plt
import linecache
import pickle
import config
import gc
import time
import sys
import re

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt='%Y-%m-%d  %H:%M:%S %a'  # 注意月份和天数不要搞乱了，这里的格式化符与time模块相同
                    )

def collate_fn(batch):
    batch = list(zip(*batch))
    return batch[0], batch[1]

def collate_fn_test(batch):
    return batch


class SummaryDataset(Dataset):
    def __init__(self, src, tgt):
        super(SummaryDataset, self).__init__()
        self.src = src
        self.tgt = tgt

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

    def __len__(self):
        return len(self.src)

class TestDataset(Dataset):
    def __init__(self, src):
        super(TestDataset, self).__init__()
        self.src = src

    def __getitem__(self, idx):
        return self.src[idx]

    def __len__(self):
        return len(self.src)


def turn2sent(sent_list):
    sentence = []
    for sent in sent_list:
        for token in sent:
            sentence.append(token)

    return sentence


def get_text(src_path, tgt_path):
    src, tgt = [], []

    with open(src_path, 'r', encoding='utf-8') as file_src:
        with open(tgt_path, 'r', encoding='utf-8') as file_tgt:

            bar = tqdm(list(zip(file_src.readlines(), file_tgt.readlines())), '读取原文文本和标签：')
            for src_line, tgt_line in bar:

                src.append(json.loads(src_line.strip()))
                tgt.append(json.loads(tgt_line.strip()))

                assert len(src[-1]) == len(tgt[-1]), "The number of sentences is not equal to the labels!"

    return src, tgt


def get_test_text(src_path):
    src = []

    with open(src_path, 'r', encoding='utf-8') as file:

        bar = tqdm(file.readlines(), '读取测试文本：')
        for line in bar:
            src.append(json.loads(line.strip()))

    return src


def get_dataloader(src_path, tgt_path, batch_size, shuffle=True):

    if tgt_path is not None:

        src, tgt = get_text(src_path, tgt_path)
        dataset = SummaryDataset(src, tgt)
        train_dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)
        return train_dataloader

    else:
        src = get_test_text(src_path)
        dataset = TestDataset(src)
        test_dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn_test)
        return test_dataloader



if __name__ == '__main__':


    loader = get_dataloader('./data/process_src_ids.txt', './data/process_labels_ids.txt', 16)

    for batch in loader:
        for item in batch[0]:

            if len(item) > 80:
                print(len(item))



