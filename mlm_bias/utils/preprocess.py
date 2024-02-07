#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import json
import time
import urllib
import pandas as pd
from mlm_bias.utils.progress import show_progress, end_progress
from mlm_bias.utils.constants import BENCHMARK_DATASET_MAP

def get_benchmarks(dataset, data_dir):
    if dataset not in BENCHMARK_DATASET_MAP.keys():
        raise Exception("Dataset Not Supported")
    url = BENCHMARK_DATASET_MAP[dataset]['download_url']
    name = BENCHMARK_DATASET_MAP[dataset]['name']
    file_name = BENCHMARK_DATASET_MAP[dataset]['file_name']
    file_path = f"{data_dir}/{file_name}"
    data_dir_exists = os.path.exists(data_dir)
    if not data_dir_exists:
        os.makedirs(data_dir)
        show_progress(1, 1, "Created Data Directory", time.time())
        end_progress()
    if not os.access(file_path, os.R_OK):
        urllib.request.urlretrieve(url, file_path)
        show_progress(1, 1, f"Downloaded Data [{name}]", time.time())
        end_progress()
    return file_path, name

def preprocess_benchmark(dataset, data_dir):
    file_path, name = get_benchmarks(dataset, data_dir)
    bias_types = []
    dis = []
    adv = []
    if dataset == 'cps':
        bdf = pd.read_csv(file_path)
        bias_types = [vls for vls in bdf['bias_type']]
        dis = [vls for vls in bdf['sent_more']]
        adv = [vls for vls in bdf['sent_less']]
    elif dataset == 'ss':
        with open(file_path, "r") as f:
            bdf = json.load(f)
            f.close()
        for bdd in bdf["data"]['intrasentence']:
            for bdds in bdd['sentences']:
                if bdds['gold_label'] == 'anti-stereotype':
                    adv.append(bdds['sentence'])
                elif bdds['gold_label'] == 'stereotype':
                    dis.append(bdds['sentence'])
                    bias_types.append(bdd['bias_type'])
    assert len(bias_types) == len(dis) == len(adv)
    show_progress(1, 1, f"Loaded Data [{name}]", time.time())
    end_progress()
    return (bias_types, dis, adv)

def preprocess_linebyline(data_dir):
    if not os.access(data_dir, os.R_OK):
        raise Exception("Can't Access Dataset")
    else:
        bias_types_path = os.path.join(data_dir, "bias_types.txt")
        dis_path = os.path.join(data_dir, "dis.txt")
        adv_path = os.path.join(data_dir, "adv.txt")
        with open(os.path.join(data_dir, "bias_types.txt"), "r") as f:
            bias_types = f.read().splitlines()
            f.close()
        with open(os.path.join(data_dir, "dis.txt"), "r") as f:
            dis = f.read().splitlines()
            f.close()
        with open(os.path.join(data_dir, "adv.txt"), "r") as f:
            adv = f.read().splitlines()
            f.close()
    assert len(bias_types) == len(dis) == len(adv)
    show_progress(1, 1, f"Loaded Data", time.time())
    return (bias_types, dis, adv)