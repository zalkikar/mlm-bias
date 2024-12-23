# Measuring Biases in Masked Language Models for PyTorch Transformers

![pypi - status](https://img.shields.io/badge/status-stable-brightgreen)
![pypi - downloads](https://img.shields.io/pypi/dm/mlm-bias)
![pypi - version](https://img.shields.io/pypi/v/mlm-bias)

Evaluate biases in pre-trained or re-trained masked language models (MLMs), such as those available through [HuggingFace](https://huggingface.co/models). This package computes bias scores across various bias types, using benchmark datasets like [CrowS-Pairs (CPS)](https://github.com/nyu-mll/crows-pairs) and [StereoSet (SS)](https://github.com/moinnadeem/StereoSet) (intrasentence), or custom datasets. You can also compare relative bias between two MLMs, or evaluate re-trained MLMs versus their pre-trained base models.

## Evaluation Methods

**Bias scores for an MLM** are computed for sentence pairs in the dataset using measures that represent MLM preference (or prediction quality). Bias against disadvantaged groups for a sentence pair is represented by a higher relative measure value for a sentence in `adv` compared to `dis`.

**Iterative Masking Experiment (IME)**: For each sentence, an MLM masks one token at a time until all tokens are masked once, generating `n` logits or predictions for a sentence with `n` tokens.

### Measures

We use state-of-the-art measures computed under the **IME**:

- **`CRR`**: Difference in reciprocal rank of a predicted token (always equal to 1) and the reciprocal rank of a masked token [arXiv](https://arxiv.org/abs/2402.13954)
- **`CRRA`**: `CRR` with Attention weights [arXiv](https://arxiv.org/abs/2402.13954)
- **&Delta;`P`**: Difference in log-liklihood of a predicted token and the masked token [arXiv](https://arxiv.org/abs/2402.13954)
- **&Delta;`PA`**: &Delta;`P` with Attention weights [arXiv](https://arxiv.org/abs/2402.13954)

Measures computed with a single encoded input (see [References](#references) for more details):
- **`CSPS`**: CrowS-Pairs Scores is a log-likelihood score for an MLM selecting unmodified tokens given modified ones [arXiv](https://arxiv.org/abs/2010.00133)
- **`SSS`**: StereoSet Score is a log-likelihood score for an MLM selecting modified tokens given unmodified ones [arXiv](https://arxiv.org/abs/2004.09456)
- **`AUL`**: All Unmasked Likelihood is a log-likelihood score generated by predicting all tokens in a single unmasked input [arXiv](https://arxiv.org/abs/2104.07496)
- **`AULA`**: `AUL` with Attention weights [arXiv](https://arxiv.org/abs/2104.07496)

*Note: Measures computed using ***IME*** take longer to compute.*

## Setup

```bash
pip install mlm-bias
```

```python
import mlm_bias

# Load the CPS dataset
cps_dataset = mlm_bias.BiasBenchmarkDataset("cps")
cps_dataset.sample(indices=list(range(10)))

# Specify the model
model = "bert-base-uncased"

# Initialize the BiasMLM evaluator
mlm_bias = mlm_bias.BiasMLM(model, cps_dataset)

# Evaluate the model
result = mlm_bias.evaluate(inc_attention=True)

# Save the results
result.save("./bert-base-uncased")
```

## Example Script

Clone the repository and install the package:

```bash
git clone https://github.com/zalkikar/mlm-bias.git
cd mlm-bias
python3 -m pip install .
```

Run the `mlm_bias.py` example script:

```bash
mlm_bias.py [-h] --data {cps,ss,custom} --model_name_or_path MODEL [--model_name_or_path_2 MODEL2] [--output OUTPUT] [--measures {all,crr,crra,dp,dpa,aul,aula,csps,sss}] [--start S] [--end E]
```

Example arguments:

```bash
# Single MLM
python3 mlm_bias.py --data cps --model_name_or_path roberta-base --start 0 --end 30
python3 mlm_bias.py --data ss --model_name_or_path bert-base-uncased --start 0 --end 30

# Relative between two MLMs
python3 mlm_bias.py --data cps --model_name_or_path roberta-base --start 0 --end 30 --model_name_or_path_2 bert-base-uncased
```

Output directories (default arguments):
- `/data` contains `cps.csv` (CPS) and/or `ss.csv` (SS).
- `/eval` contains `out.txt` with computed bias scores and pickled result objects.


### Example Output:

```bash
python3 mlm_bias.py --data cps --model_name_or_path bert-base-uncased --start 0 --end 30
```

```bash
Created output directory.
Created Data Directory |██████████████████████████████| 1/1 [100%] in 0s ETA: 0s
Downloaded Data [CrowSPairs] |██████████████████████████████| 1/1 [100%] in 0s ETA: 0s
Loaded Data [CrowSPairs] |██████████████████████████████| 1/1 [100%] in 0s ETA: 0s
Evaluating Bias [bert-base-uncased] |██████████████████████████████| 30/30 [100%] in 1m 4s ETA: 0s
Saved bias results for bert-base-uncased in ./eval/bert-base-uncased
Saved scores in ./eval/out.txt
--------------------------------------------------
MLM: bert-base-uncased
CRR total = 26.667
CRRA total = 30.0
ΔP total = 46.667
ΔPA total = 43.333
AUL total = 36.667
AULA total = 40.0
SSS total = 30.0
CSPS total = 33.333
```

## Custom Datasets

Compute bias scores for a custom dataset directory with the following line-by-line files:

- `bias_types.txt` containing bias categories.
- `dis.txt` and `adv.txt` containing sentence pairs, where:
  - `dis.txt` contains sentences with bias against disadvantaged groups (stereotypical) and
  - `adv.txt` contains sentences with bias against advantaged groups (anti-stereotypical).

## Citation

If using this for research, please cite the following:

```bibtex
@misc{zalkikar2024measuringsocialbiasesmasked,
      title={Measuring Social Biases in Masked Language Models by Proxy of Prediction Quality},
      author={Rahul Zalkikar and Kanchan Chandra},
      year={2024},
      eprint={2402.13954},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.13954}
}
```

## References

```bibtex
@article{Kaneko_Bollegala_2022,
      title={Unmasking the Mask – Evaluating Social Biases in Masked Language Models},
      volume={36},
      url={https://ojs.aaai.org/index.php/AAAI/article/view/21453},
      DOI={10.1609/aaai.v36i11.21453},
      number={11},
      journal={Proceedings of the AAAI Conference on Artificial Intelligence},
      author={Kaneko, Masahiro and Bollegala, Danushka},
      year={2022},
      month={Jun.},
      pages={11954-11962}
}
```

```bibtex
@InProceedings{10.1007/978-3-031-33374-3_42,
      author="Salutari, Flavia
        and Ramos, Jerome
        and Rahmani, Hossein A.
        and Linguaglossa, Leonardo
        and Lipani, Aldo",
      editor="Kashima, Hisashi
        and Ide, Tsuyoshi
        and Peng, Wen-Chih",
      title="Quantifying the Bias of Transformer-Based Language Models for African American English in Masked Language Modeling",
      booktitle="Advances in Knowledge Discovery and Data Mining",
      year="2023",
      publisher="Springer Nature Switzerland",
      address="Cham",
      pages="532--543",
      isbn="978-3-031-33374-3"
}
```

```bibtex
@inproceedings{nangia-etal-2020-crows,
      title = "{C}row{S}-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models",
      author = "Nangia, Nikita  and
        Vania, Clara  and
        Bhalerao, Rasika  and
        Bowman, Samuel R.",
      editor = "Webber, Bonnie  and
        Cohn, Trevor  and
        He, Yulan  and
        Liu, Yang",
      booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
      month = nov,
      year = "2020",
      address = "Online",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/2020.emnlp-main.154",
      doi = "10.18653/v1/2020.emnlp-main.154",
      pages = "1953--1967"
}
```

```bibtex
@inproceedings{nadeem-etal-2021-stereoset,
      title = "{S}tereo{S}et: Measuring stereotypical bias in pretrained language models",
      author = "Nadeem, Moin  and
        Bethke, Anna  and
        Reddy, Siva",
      editor = "Zong, Chengqing  and
        Xia, Fei  and
        Li, Wenjie  and
        Navigli, Roberto",
      booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
      month = aug,
      year = "2021",
      address = "Online",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/2021.acl-long.416",
      doi = "10.18653/v1/2021.acl-long.416",
      pages = "5356--5371"
}
```
