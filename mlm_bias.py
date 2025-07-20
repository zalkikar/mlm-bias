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

def format_mlm_bias_results(result, total_only=False, sep="\n"):
    outl = []
    if total_only:
        for measure in result['bias_scores'].keys():
            outl.append((f"{measure.replace('d','Δ').upper()} " \
                         f"total = {round(result['bias_scores'][measure]['total'],3)}"))
    else:
        for measure in result['bias_scores'].keys():
            outl.append(f"Measure = {measure.replace('d','Δ').upper()}")
            for bias_type, score in result['bias_scores'][measure].items():
                outl.append(f"- {bias_type} = {round(score,3)}")
    return f"{sep}".join(outl)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data',
                        type=str,
                        required=True,
                        help=('Paired sentences from benchmark or supplied line by line dataset in /data directory. '+
                              'Provide bias types in "<data>/bias_types.txt" and biased sentences in "<data>/dis.txt" and "<data>/adv.txt" accordingly.'),
                        choices=['cps','ss','custom'])

    parser.add_argument('--model_name_or_path',
                        type=str,
                        required=True,
                        help=('Model (MLM) to compute bias measures for. '+
                              'Must be supported by HuggingFace.'))

    parser.add_argument('--model_name_or_path_2',
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

    parser.add_argument('--start',
                        type=int,
                        required=False,
                        default=-1,
                        help='Start index of dataset sample.')

    parser.add_argument('--end',
                        type=int,
                        required=False,
                        default=-1,
                        help='End index of dataset sample.')

    parser.add_argument('--batching',
                        action='store_true',
                        help='Batched inputs.')

    args = parser.parse_args()

    try:
        outDirExists = os.path.exists(os.path.dirname(args.output))
        if not outDirExists:
            os.makedirs(os.path.dirname(args.output))
            print("Created output directory.")
    except Exception as ex:
        raise Exception(f"Could not create output directory {args.output}\n{ex}")
    try:
        if args.data == 'custom':
            dataset = BiasLineByLineDataset(args.data)
        else:
            dataset = BiasBenchmarkDataset(args.data)
    except Exception as ex:
        raise Exception(f"Could not load dataset {args.data}\n{ex}")
    if args.start != -1 and args.start < 0:
        raise argparse.ArgumentTypeError(f"{args.start} is not a positive integer")
    if args.end != -1 and (args.end < 0 or args.end <= args.start):
        raise argparse.ArgumentTypeError(f"{args.end} is not a valid positive integer greater than {args.start}")

    if args.start != -1 and args.end != -1:
        dataset.sample(indices=list(range(args.start, args.end)))

    output_dir = os.path.dirname(args.output)

    out = ""
    model_name_or_path = args.model_name_or_path
    try:
        model_bias = BiasMLM(args.model_name_or_path, dataset)
    except Exception as ex:
        raise Exception(f"Could not load {args.model_name_or_path}\n{ex}")
    if args.measures == 'all':
        res1 = model_bias.evaluate(attention=True, batching=args.batching)
    else:
        res1 = model_bias.evaluate(measures=args.measures, attention=True, batching=args.batching)
    output_dir_res1 = os.path.join(output_dir, res1['model_name_or_path'])
    res1.save(output_dir_res1)
    print(f"Saved bias results for {res1['model_name_or_path']} in {output_dir_res1}")
    out += "-"*50
    out += "\n"
    out += f"MLM: {res1['model_name_or_path']}"
    out += "\n"
    out += format_mlm_bias_results(res1)

    res2 = None
    if args.model_name_or_path_2 != "":
        model = args.model_name_or_path_2
        model_bias = BiasMLM(args.model_name_or_path_2, dataset)
        if args.measures == 'all':
            res2 = model_bias.evaluate(attention=True, batching=args.batching)
        else:
            res2 = model_bias.evaluate(measures=args.measures, attention=True, batching=args.batching)
        output_dir_res2 = os.path.join(output_dir, res2['model_name_or_path'])
        res2.save(output_dir_res2)
        print(f"Saved bias results for {res2['model_name_or_path']} in {output_dir_res2}")
        out += "-"*50
        out += "\n"
        out += f"MLM: {res2['model_name_or_path']}"
        out += "\n"
        out += format_mlm_bias_results(res2)

    if res2 is not None:
        mlm_bias_relative = RelativeBiasMLMs(res1, res2)
        res3 = mlm_bias_relative.evaluate()
        output_dir_res3 = os.path.join(output_dir, f"{res1['model_name_or_path']}_{res2['model_name_or_path']}")
        res3.save(output_dir_res3)
        print(f"Saved bias results for {res1['model_name_or_path']} relative to {res2['model_name_or_path']} in {output_dir_res3}")
        out += "-"*50
        out += "\n"
        out += f"Relative {res1['model_name_or_path']}, {res2['model_name_or_path']}"
        out += "\n"
        out += format_mlm_bias_results(res3)

    with open(args.output, 'w+', encoding='utf-8') as f:
        f.write(out)

    print(f"Saved scores in {args.output}")

    out = "-"*50
    out += "\n"
    out += f"MLM: {res1['model_name_or_path']}"
    out += "\n"
    out += format_mlm_bias_results(res1, total_only=True)
    print(out)
