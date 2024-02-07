#!/usr/bin/python
# -*- coding: utf-8 -*-

BENCHMARK_DATASET_MAP = {
    "cps": {
        'name': "CrowSPairs",
        'file_name': 'cps.csv',
        'download_url': "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv",
    },
    "ss": {
        'name': "StereoSet",
        'file_name': 'ss.json',
        'download_url': "https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json",
    },
}
SUPPORTED_MEASURES = ['crr','crra','dp','dpa','aul','aula','sss','csps']
SUPPORTED_MEASURES_ATTENTION = ['crr','dp','aul']