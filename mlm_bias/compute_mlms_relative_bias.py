#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from typing import Optional
from mlm_bias.bias_results import BiasResults, RelativeBiasResults

class RelativeBiasMLMs():
    """
    Class for computing biases for an MLM relative to another.
    """

    def __init__(
        self,
        mlm1_bias_results: BiasResults,
        mlm2_bias_results: BiasResults
    ):
        measures_common = set(mlm1_bias_results["measures"]) & set(mlm2_bias_results["measures"])
        assert measures_common.issubset(mlm2_bias_results["measures"])
        assert measures_common.issubset(mlm1_bias_results["measures"])
        assert (
            str(mlm1_bias_results["eval_results"]["bias_types"]) ==
            str(mlm2_bias_results["eval_results"]["bias_types"])
        )
        self.results = RelativeBiasResults()
        self.bias_types = mlm1_bias_results["eval_results"]["bias_types"]
        self.unique_biases = list(set(self.bias_types))
        self.measures = measures_common
        self.mlm1_bias_results = mlm1_bias_results
        self.mlm2_bias_results = mlm2_bias_results

    def __getitem__(self, key):
        return getattr(self, key)

    def evaluate(self, measures: Optional[list] = None):
        if measures is not None:
            assert set(measures).issubset(self.measures)
        else:
            measures = self.measures
        self.bias_scores = {m:{} for m in measures}
        for m in measures:
            m1_s1 = self.mlm1_bias_results.eval(1, m)
            m1_s2 = self.mlm1_bias_results.eval(2, m)
            m2_s1 = self.mlm2_bias_results.eval(1, m)
            m2_s2 = self.mlm2_bias_results.eval(2, m)
            assert len(m1_s1) == len(m2_s2) == len(m1_s2) == len(m2_s1) == len(self.bias_types)
            mdifs = [((m21 - m11) - (m22 - m21)) for m22, m21, m12, m11 in zip(m2_s2, m2_s1, m1_s2, m1_s1)]
            self.bias_scores[m]['total'] = 100 * np.mean([1 if mdif > 0 else 0 for mdif in mdifs])
            for b in self.unique_biases:
                m1_s1 = self.mlm1_bias_results.eval(1, m, b)
                m1_s2 = self.mlm1_bias_results.eval(2, m, b)
                m2_s1 = self.mlm2_bias_results.eval(1, m, b)
                m2_s2 = self.mlm2_bias_results.eval(2, m, b)
                mdifs = [((m21 - m11) - (m22 - m21)) for m22, m21, m12, m11 in zip(m2_s2, m2_s1, m1_s2, m1_s1)]
                self.bias_scores[m][b] = 100 * np.mean([1 if mdif > 0 else 0 for mdif in mdifs])
        self.results(
            self.mlm1_bias_results["model_name"],
            self.mlm2_bias_results["model_name"],
            measures,
            self.bias_scores
        )
        return self.results