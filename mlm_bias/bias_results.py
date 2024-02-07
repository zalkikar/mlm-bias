#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
from typing import Optional

class BiasResults():

    model_name = None
    measures = None
    eval_results = None
    bias_scores = None

    def __call__(
        self,
        model_name: str,
        measures: list,
        eval_results: dict,
        bias_scores: dict,
    ):
        self.model_name = model_name
        self.measures = measures
        self.eval_results = eval_results
        self.bias_scores = bias_scores

    def __getitem__(self, key):
        return getattr(self, key)

    def save(self, file_path: Optional[str] = None):
        if file_path is None:
            fp = f'{self.model_name}.bias'
        else:
            fp = file_path
        with open(fp, 'wb') as f:
            f.write(pickle.dumps(self.__dict__))
            f.close()

    def load(self, file_path: Optional[str] = None):
        if file_path is None:
            fp = f'{self.model_name}.bias'
        else:
            fp = file_path
        with open(fp, 'rb') as f:
            data = pickle.load(f)
            f.close()
        self.model_name = data['model_name']
        self.measures = data['measures']
        self.eval_results = data['eval_results']
        self.bias_scores = data['bias_scores']

    def eval(
        self,
        set_num: int,
        measure: str,
        bias_type: Optional[str] = None
    ):
        eval_res = self.eval_results[f'S{set_num}'][measure]
        if bias_type is not None:
            return [
                v for vi,v in enumerate(eval_res)
                if self.eval_results['bias_types'][vi] == bias_type
            ]
        return eval_res

class RelativeBiasResults():

    model1_name = None
    model2_name = None
    measures = None
    bias_scores = None

    def __call__(
        self,
        model1_name: str,
        model2_name: str,
        measures: list,
        bias_scores: dict,
    ):
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.measures = measures
        self.bias_scores = bias_scores

    def __getitem__(self, key):
        return getattr(self, key)

    def save(self, file_path: Optional[str] = None):
        if file_path is None:
            fp = f'{self.model1_name}_{self.model2_name}.bias'
        else:
            fp = file_path
        with open(fp, 'wb') as f:
            f.write(pickle.dumps(self.__dict__))
            f.close()

    def load(self, file_path: Optional[str] = None):
        if file_path is None:
            fp = f'{self.model1_name}_{self.model2_name}.bias'
        else:
            fp = file_path
        with open(fp, 'rb') as f:
            data = pickle.load(f)
            f.close()
        self.model1_name = data['model1_name']
        self.model2_name = data['model2_name']
        self.measures = data['measures']
        self.bias_scores = data['bias_scores']