#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Optional
from mlm_bias.utils.preprocess import preprocess_benchmark, preprocess_linebyline

class BiasDataset():
    def __init__(self, bias_types: list, dis: list, adv: list):
        assert len(dis) == len(adv) == len(bias_types)
        self.bias_types_all = bias_types
        self.dis_all = dis
        self.adv_all = adv
        self.reset()

    def __len__(self):
        return len(self.bias_types)

    def __getitem__(self, idx: int):
        return self.bias_types[idx], self.dis[idx], self.adv[idx]

    def reset(self):
        self.bias_types = self.bias_types_all
        self.dis = self.dis_all
        self.adv = self.adv_all

    def sample(self, indices: list):
        self.bias_types = [self.bias_types_all[i] for i in indices]
        self.dis = [self.dis_all[i] for i in indices]
        self.adv = [self.adv_all[i] for i in indices]

class BiasBenchmarkDataset(BiasDataset):
    def __init__(self, dataset: str, data_dir: Optional[str] = './data'):
        self.bias_types_all, self.dis_all, self.adv_all = preprocess_benchmark(dataset, data_dir)
        self.reset()

class BiasLineByLineDataset(BiasDataset):
    def __init__(self, data_dir: Optional[str] = './data'):
        self.bias_types_all, self.dis_all, self.adv_all = preprocess_linebyline(data_dir)
        self.reset()