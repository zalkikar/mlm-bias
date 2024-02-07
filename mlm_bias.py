#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import argparse
from mlm_bias import (
    BiasBenchmarkDataset,
    BiasLineByLineDataset,
    BiasMLM,
    RelativeBiasMLMs
)

def pretty_print(res, out, m_name, sep="\n"):
    out += ('-'*50)
    out += sep
    out += (f"MLM: {m_name}")
    out += sep
    for measure in res['bias_scores'].keys():
        out += (f"Measure = {measure.replace('d','Δ').upper()}")
        out += sep
        for bias_type, score in res['bias_scores'][measure].items():
            out += (f"- {bias_type} = {round(score,3)}")
            out += sep
    return out

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data',
                        type=str,
                        required=True,
                        help=('Paired sentences from benchmark or supplied line by line dataset in /data directory. '+
                              'Provide bias types in "<data>/bias_types.txt" and biased sentences in "<data>/dis.txt" and "<data>/adv.txt" accordingly.'),
                        choices=['cps','ss','custom'])

    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help=('Model (MLM) to compute bias measures for. '+
                              'Must be supported by HuggingFace.'))

    parser.add_argument('--model2',
                        type=str,
                        required=False,
                        default="",
                        help=('Model (MLM) to compute bias measures for. '+
                              'Must be supported by HuggingFace. '+
                              'Used to compare with "--model"'))

    parser.add_argument('--output',
                        type=str,
                        required=False,
                        default="./eval/out.txt",
                        help='Full path (eg. dir/file.txt) for output directory with computed measures.')

    parser.add_argument('--measures',
                        type=str,
                        required=False,
                        default='all',
                        help='Measures computed to evaluate bias in MLMs.',
                        choices=['all','crr','crra','dp','dpa','aul','aula','csps','sss'])

    parser.add_argument('--s',
                        type=int,
                        required=False,
                        default=-1,
                        help='Start index of dataset sample.')

    parser.add_argument('--e',
                        type=int,
                        required=False,
                        default=-1,
                        help='End index of dataset sample.')

    args = parser.parse_args()

    outDirExists = os.path.exists(os.path.dirname(args.output))
    if not outDirExists:
        os.makedirs(os.path.dirname(args.output))
        print("Created output directory.")

    if args.data == 'custom':
        dataset = BiasLineByLineDataset(args.data)
    else:
        dataset = BiasBenchmarkDataset(args.data)

    if args.s != -1 and args.e != -1:
        dataset.sample(indices=list(range(args.s, args.e)))

    output_dir = os.path.dirname(args.output)

    out = ""
    model = args.model
    model_bias = BiasMLM(args.model, dataset)
    if args.measures == 'all':
        res1 = model_bias.evaluate(inc_attention=True)
    else:
        res1 = model_bias.evaluate(measures=args.measures, inc_attention=True)
    output_dir_res1 = os.path.join(output_dir, res1['model_name'])
    res1.save(output_dir_res1)
    print(f"Saved bias results for {res1['model_name']} in {output_dir_res1}")
    out = pretty_print(res1, out, m_name=res1['model_name'])

    res2 = None
    if args.model2 != "":
        model = args.model2
        model_bias = BiasMLM(args.model2, dataset)
        if args.measures == 'all':
            res2 = model_bias.evaluate(inc_attention=True)
        else:
            res2 = model_bias.evaluate(measures=args.measures, inc_attention=True)
        output_dir_res2 = os.path.join(output_dir, res2['model_name'])
        res2.save(output_dir_res2)
        print(f"Saved bias results for {res2['model_name']} in {output_dir_res2}")
        out = pretty_print(res2, out, m_name=res2['model_name'])

    if res2 is not None:
        mlm_bias_relative = RelativeBiasMLMs(res1, res2)
        res3 = mlm_bias_relative.evaluate()
        output_dir_res3 = os.path.join(output_dir, f"{res1['model_name']}_{res2['model_name']}")
        res3.save(output_dir_res3)
        print(f"Saved bias results for {res1['model_name']} relative to {res2['model_name']} in {output_dir_res3}")
        out = pretty_print(res3, out, m_name=f"Relative {res1['model_name']}, {res2['model_name']}")

    with open(args.output, 'w+', encoding='utf-8') as f:
        f.write(out)

    print(f"Saved scores in {args.output}")
