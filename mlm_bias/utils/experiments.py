#!/usr/bin/python
# -*- coding: utf-8 -*-

import difflib
import regex as re

def get_mask_combinations(sent, tokenizer, skip_space=False, rm_punc=True):
    sent_toks = []
    gt = []
    mask_ind = 0
    if rm_punc:
        sent = ' '.join(re.sub('[^A-Za-z0-9 _\-]+', '', sent).split())
    sent_enc = tokenizer.encode(sent, add_special_tokens=False)
    mask_tok_id = tokenizer.mask_token_id
    for ind,tokid in enumerate(sent_enc):
        if skip_space and tokid == tokenizer.encode(' ', add_special_tokens=False)[0]:
            continue
        sent_toks.append([s if sind != ind else mask_tok_id for sind,s in enumerate(sent_enc)])
        gt.append(tokid)
    return {'sent':sent_toks, 'gt':gt}

def get_span(seq1, seq2, operation):
    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        if (operation == 'equal' and op[0] == 'equal') \
                or (operation == 'diff' and op[0] != 'equal'):
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]
    return template1, template2