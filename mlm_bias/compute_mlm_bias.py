#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import torch
import numpy as np
from typing import Optional
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from mlm_bias.bias_datasets import BiasDataset
from mlm_bias.bias_results import BiasResults
from mlm_bias.utils import (
    compute_aul,
    compute_crr_dp,
    compute_crr_dp_batched,
    compute_csps,
    compute_sss,
    end_progress,
    get_mask_combinations,
    get_span,
    show_progress,
    SUPPORTED_MEASURES,
    SUPPORTED_MEASURES_ATTENTION
)

class BiasMLM():
    """
    Class for computing biases for an MLM.
    """

    def __init__(
        self,
        model_name_or_path: str,
        dataset: BiasDataset,
        device: Optional[str] = None,
    ):
        self.results = BiasResults()
        self.dataset = dataset
        self.model_name_or_path = model_name_or_path
        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            output_hidden_states=True,
            output_attentions=True,
            attn_implementation="eager")
        self.model = AutoModelForMaskedLM.from_config(self.model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.mask_id = self.tokenizer.mask_token_id
        self.device = None
        if device is not None:
            if not torch.cuda.is_available() and device == "cuda":
                raise Exception("Cuda Not Available")
            self.device = device
            self.model.to(self.device)
        self.model.eval()
        test_special = self.tokenizer.encode("test", add_special_tokens=True, return_tensors="pt")
        self.special_tok_s = test_special[0].tolist()[0]
        self.special_tok_e = test_special[0].tolist()[-1]
        self.supported_measures = SUPPORTED_MEASURES
        self.supported_measures_attention = SUPPORTED_MEASURES_ATTENTION

    def __getitem__(self, key):
        return getattr(self, key)

    def scores(self):
        total = 0
        unq_bias_types = list(set(self.eval_results["bias_types"]))
        bias_total =  {b: 0 for b in unq_bias_types}
        self.bias_scores = {m: {b: 0 for b in unq_bias_types} for m in self.measures}
        for m_ind, m in enumerate(self.measures):
            m1s = self.eval_results["S1"][m]
            m2s = self.eval_results["S2"][m]
            if "total" not in self.bias_scores[m]:
                self.bias_scores[m]["total"] = 0
            for ind,(m1,m2) in enumerate(zip(m1s, m2s)):
                bias_type = self.eval_results["bias_types"][ind]
                if m_ind == 0:
                    total += 1
                    bias_total[bias_type] += 1
                if m in ["crr","dp","crra","dpa"]:
                    if m2 > m1:
                        self.bias_scores[m]["total"] += 1
                        self.bias_scores[m][bias_type] += 1
                else:
                    if m1 > m2:
                        self.bias_scores[m]["total"] += 1
                        self.bias_scores[m][bias_type] += 1
        if total > 0:
            for m in self.measures:
                self.bias_scores[m]["total"] = (100*(self.bias_scores[m]["total"]/total))
                for b in unq_bias_types:
                    if bias_total[b] > 0:
                        self.bias_scores[m][b] = (100*(self.bias_scores[m][b]/bias_total[b]))

    def evaluate(
        self,
        measures: Optional[list] = None,
        attention: Optional[bool] = True,
        batching: bool = True,
    ):
        assert self.model.training == False
        if measures is None:
            measures = self.supported_measures
        else:
            for m in measures:
                if m not in self.supported_measures:
                    raise Exception("Measure Not Supported")
            if attention:
                measures_temp = measures
                measures.extend([
                    m+"a" for m in measures_temp
                    if m in self.supported_measures_attention
                ])
                del measures_temp
                measures = list(set(measures))

        mm = []
        if "crr" in measures:
            mm.append("crr")
        if "crra" in measures:
            mm.append("crra")
        if "dp" in measures:
            mm.append("dp")
        if "dpa" in measures:
            mm.append("dpa")

        self.eval_results = {
            "bias_types": [],
            "S1": {m:[] for m in measures},
            "S2": {m:[] for m in measures},
        }

        start_time = time.time()
        for index in range(len(self.dataset)):
            show_progress(index, len(self.dataset), f"Evaluating Bias [{self.model_name_or_path}]", start_time)
            bias_type, s1, s2 = self.dataset[index]
            self.eval_results["bias_types"].append(bias_type)
            if "crr" in measures or "dp" in measures:
                utterance_measures = {
                    "S1": {m:[] for m in mm},
                    "S2": {m:[] for m in mm}
                }
                if batching:
                    groups_all = []
                    ground_truths_all = []
                    token_ids_all = []
                    mask_token_index_all = []
                    for i, sent_text in enumerate([s1, s2]):
                        sent_data = get_mask_combinations(sent_text, tokenizer=self.tokenizer)
                        for st, gt in zip(sent_data["sent"], sent_data["gt"]):
                            token_ids = torch.tensor([self.special_tok_s] + st + [self.special_tok_e])
                            mask_token_index = np.where(np.array(token_ids) == self.mask_id)[0][0]
                            groups_all.append(i+1)
                            ground_truths_all.append(gt)
                            token_ids_all.append(token_ids)
                            mask_token_index_all.append(mask_token_index)
                    token_ids_all = torch.nn.utils.rnn.pad_sequence(
                        token_ids_all,
                        batch_first=True,
                        padding_value=self.tokenizer.pad_token_id
                    ).to(self.device)
                    attention_mask = (token_ids_all != self.tokenizer.pad_token_id).long()
                    mjs = compute_crr_dp_batched(
                        self.model,
                        token_ids_all,
                        mask_token_index_all,
                        attention_mask,
                        ground_truths_all,
                        measures=measures,
                        attention=attention,
                        log_softmax=False
                    )
                    for i, mj in enumerate(mjs):
                        for m in mm:
                            utterance_measures[f"S{groups_all[i]}"][m].append(mj["masked_token"][m])
                else:
                    for i, sent_text in enumerate([s1, s2]):
                        sent_data = get_mask_combinations(sent_text, tokenizer=self.tokenizer)
                        for st, gt in zip(sent_data["sent"], sent_data["gt"]):
                            token_ids = [self.special_tok_s]
                            token_ids.extend(st)
                            token_ids.append(self.special_tok_e)
                            token_ids = torch.tensor([token_ids])
                            mask_token_index = np.where(np.array(token_ids) == self.mask_id)[1]
                            if self.device is not None:
                                token_ids.to(self.device)
                            mj = compute_crr_dp(
                                self.model,
                                token_ids,
                                mask_token_index,
                                gt,
                                measures=measures,
                                attention=attention,
                                log_softmax=False
                            )
                            for m in mm:
                                utterance_measures[f"S{i+1}"][m].append(mj["masked_token"][m])
                for m in mm:
                    self.eval_results["S1"][m].append(np.mean(utterance_measures["S1"][m]))
                    self.eval_results["S2"][m].append(np.mean(utterance_measures["S2"][m]))
            if "aul" in measures or "csps" in measures or "sss" in measures:
                token_ids_dis = self.tokenizer.encode(s1, return_tensors="pt")
                token_ids_adv = self.tokenizer.encode(s2, return_tensors="pt")
                if "aul" in measures:
                    mj_dis = compute_aul(self.model, token_ids_dis, attention=attention, log_softmax=True)
                    mj_adv = compute_aul(self.model, token_ids_adv, attention=attention, log_softmax=True)
                    self.eval_results[f"S1"]["aul"].append(mj_dis["aul"])
                    self.eval_results[f"S2"]["aul"].append(mj_adv["aul"])
                    if attention:
                        self.eval_results[f"S1"]["aula"].append(mj_dis["aula"])
                        self.eval_results[f"S2"]["aula"].append(mj_adv["aula"])
                if "csps" in measures:
                    dis_spans, adv_spans = get_span(token_ids_dis[0], token_ids_adv[0], "equal")
                    mj_dis = compute_csps(self.model, token_ids_dis, dis_spans, self.mask_id, log_softmax=True)
                    mj_adv = compute_csps(self.model, token_ids_adv, adv_spans, self.mask_id, log_softmax=True)
                    self.eval_results[f"S1"]["csps"].append(mj_dis["csps"])
                    self.eval_results[f"S2"]["csps"].append(mj_adv["csps"])
                if "sss" in measures:
                    dis_spans, adv_spans = get_span(token_ids_dis[0], token_ids_adv[0], "diff")
                    mj_dis = compute_sss(self.model, token_ids_dis, dis_spans, self.mask_id, log_softmax=True)
                    mj_adv = compute_sss(self.model, token_ids_adv, adv_spans, self.mask_id, log_softmax=True)
                    self.eval_results[f"S1"]["sss"].append(mj_dis["sss"])
                    self.eval_results[f"S2"]["sss"].append(mj_adv["sss"])
        show_progress(index+1, len(self.dataset), f"Evaluating Bias [{self.model_name_or_path}]", start_time)
        end_progress()
        self.measures = measures
        self.scores()
        self.results(
            self.model_name_or_path,
            self.measures,
            self.eval_results,
            self.bias_scores
        )
        return self.results
