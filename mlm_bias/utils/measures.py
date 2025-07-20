#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import numpy as np

def get_mlm_output(model, inputs, attention_mask=None):
    with torch.no_grad():
        output = model(inputs, attention_mask=attention_mask, return_dict=True)
    return output

@torch.no_grad()
def compute_crr_dp(
    model,
    token_ids,
    mask_token_index,
    masked_tok,
    measures=["crr","dp"],
    attention=True,
    log_softmax=False
):
    f_softmax = torch.nn.LogSoftmax(dim=1) if log_softmax else torch.nn.Softmax(dim=1)
    output = get_mlm_output(model, token_ids)
    token_logits = output.logits
    if attention:
        attn = output.attentions
        attentions = torch.mean(torch.cat(attn,0),0)
        attentions_avg = torch.mean(attentions,0)
        attentions_avg_tok = torch.mean(attentions_avg,0)[1:-1]
        attw = np.mean([att.tolist() for att in [attentions_avg_tok]], axis=1)
    mask_token_logits = token_logits.squeeze(0)[mask_token_index]
    mask_token_probs = f_softmax(mask_token_logits)
    top_toks = torch.topk(mask_token_probs, mask_token_probs.shape[1], dim=1)
    top_toks = top_toks.indices[0].tolist()
    top_token = top_toks[0]
    top_token_score = mask_token_probs[:, top_token].tolist()[0]
    tok_inds = list(range(mask_token_probs.shape[1]))
    masked_token_index = tok_inds.index(masked_tok)
    masked_token_score = mask_token_probs[:, masked_token_index].tolist()[0]
    masked_token_rank = top_toks.index(masked_tok) + 1
    token_j = {
        "token_id": masked_tok,
        "score": masked_token_score,
        "rank": masked_token_rank
    }
    if "crr" in measures:
        token_j["crr"] = (1 - (1/masked_token_rank))
        if attention:
            token_j["crra"] = (attw[0] * (1 - np.log(1/masked_token_rank)))
    if "dp" in measures:
        token_j["dp"] = (np.log(top_token_score) - np.log(masked_token_score))
        if attention:
            token_j["dpa"] = (attw[0] * (np.log(top_token_score) - np.log(masked_token_score)))
    return {
        "prediction": {
            "token_id": top_token,
            "score": top_token_score,
            "rank": 1
        },
        "masked_token": token_j
    }

@torch.no_grad()
def compute_crr_dp_batched(
    model,
    token_ids,
    mask_token_indexes,
    attention_mask,
    masked_toks,
    measures=["crr","dp"],
    attention=True,
    log_softmax=False
):
    f_softmax = torch.nn.LogSoftmax(dim=1) if log_softmax else torch.nn.Softmax(dim=1)
    output = get_mlm_output(model, token_ids, attention_mask)
    token_logits = output.logits
    if attention:
        attn = output.attentions
    results = []
    for i in range(token_ids.size(0)):
        if attention:
            attentions = torch.mean(torch.cat([layer[i:i+1] for layer in attn],0),0)
            attentions_avg = torch.mean(attentions,0)
            attentions_avg_tok = torch.mean(attentions_avg,0)[1:-1]
            attw = np.mean([att.tolist() for att in [attentions_avg_tok]], axis=1)
        mask_token_index = mask_token_indexes[i]
        masked_tok = masked_toks[i]
        mask_token_logits = token_logits[i, mask_token_index, :]
        mask_token_probs = f_softmax(mask_token_logits.unsqueeze(0))
        top_toks = torch.topk(mask_token_probs, mask_token_probs.shape[1], dim=1)
        top_toks = top_toks.indices[0].tolist()
        top_token = top_toks[0]
        top_token_score = mask_token_probs[:, top_token].tolist()[0]
        tok_inds = list(range(mask_token_probs.shape[1]))
        masked_token_index = tok_inds.index(masked_tok)
        masked_token_score = mask_token_probs[:, masked_token_index].tolist()[0]
        masked_token_rank = top_toks.index(masked_tok) + 1
        token_j = {
            "token_id": masked_tok,
            "score": masked_token_score,
            "rank": masked_token_rank
        }
        if "crr" in measures:
            token_j["crr"] = (1 - (1/masked_token_rank))
            if attention:
                token_j["crra"] = (attw[0] * (1 - np.log(1/masked_token_rank)))
        if "dp" in measures:
            token_j["dp"] = (np.log(top_token_score) - np.log(masked_token_score))
            if attention:
                token_j["dpa"] = (attw[0] * (np.log(top_token_score) - np.log(masked_token_score)))
        results.append({
            "prediction": {
                "token_id": top_token,
                "score": top_token_score,
                "rank": 1
            },
            "masked_token": token_j
        })
    return results

@torch.no_grad()
def compute_aul(model, token_ids, attention=True, log_softmax=True):
    f_softmax = torch.nn.LogSoftmax(dim=1) if log_softmax else torch.nn.Softmax(dim=1)
    output = get_mlm_output(model, token_ids)
    logits = output.logits.squeeze(0)
    probs = f_softmax(logits)
    token_ids = token_ids.view(-1, 1).detach()
    token_probs = probs.gather(1, token_ids)[1:-1]
    if attention:
        attentions = torch.mean(torch.cat(output.attentions, 0), 0)
        averaged_attentions = torch.mean(attentions, 0)
        averaged_token_attentions = torch.mean(averaged_attentions, 0)
        token_probs_attns = token_probs.squeeze(1) * averaged_token_attentions[1:-1]
    sentence_log_prob = torch.mean(token_probs)
    sentence_log_prob_attns = torch.mean(token_probs_attns)
    score = sentence_log_prob.item()
    score_attns = sentence_log_prob_attns.item()
    sorted_indexes = torch.sort(probs, dim=1, descending=True)[1]
    ranks = torch.where(sorted_indexes == token_ids)[1] + 1
    ranks = ranks.tolist()
    return {
        "aul": score,
        "aula": score_attns,
        "ranks": ranks
    }

@torch.no_grad()
def compute_csps(model, token_ids, spans, mask_id, log_softmax=True):
    f_softmax = torch.nn.LogSoftmax(dim=1) if log_softmax else torch.nn.Softmax(dim=1)
    spans = spans[1:-1]
    masked_token_ids = token_ids.repeat(len(spans), 1)
    masked_token_ids[range(masked_token_ids.size(0)), spans] = mask_id
    hidden_states = get_mlm_output(model, masked_token_ids)
    hidden_states = hidden_states[0]
    token_ids = token_ids.view(-1)[spans]
    probs = f_softmax(hidden_states[range(hidden_states.size(0)), spans, :])
    span_probs = probs[range(hidden_states.size(0)), token_ids]
    score = torch.sum(span_probs).item()
    sorted_indexes = torch.sort(probs, dim=1, descending=True)[1]
    ranks = torch.where(sorted_indexes == token_ids.view(-1, 1))[1] + 1
    ranks = ranks.tolist()
    return {
        "csps": score,
        "ranks": ranks
    }

@torch.no_grad()
def compute_sss(model, token_ids, spans, mask_id, log_softmax=True):
    f_softmax = torch.nn.LogSoftmax(dim=1) if log_softmax else torch.nn.Softmax(dim=1)
    masked_token_ids = token_ids.clone()
    masked_token_ids[:, spans] = mask_id
    hidden_states = get_mlm_output(model, masked_token_ids)
    hidden_states = hidden_states[0].squeeze(0)
    token_ids = token_ids.view(-1)[spans]
    probs = f_softmax(hidden_states)[spans]
    span_probs = probs[:,token_ids]
    score = torch.mean(span_probs).item()
    if probs.size(0) != 0:
        sorted_indexes = torch.sort(probs, dim=1, descending=True)[1]
        ranks = torch.where(sorted_indexes == token_ids.view(-1, 1))[1] + 1
        ranks = ranks.tolist()
    else:
        ranks = [-1]
    return {
        "sss": score,
        "ranks": ranks
    }