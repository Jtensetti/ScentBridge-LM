import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('e1',Path(__file__).resolve().parents[1]/'experiments/e1_paired_retrieval.py')
e1=importlib.util.module_from_spec(spec);spec.loader.exec_module(e1)

class ContrastiveTests(unittest.TestCase):
    def test_gradients_against_central_difference(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(4,6));y=rng.normal(size=(4,5))
        a=rng.normal(size=(6,3));b=rng.normal(size=(5,3))
        _,ga,gb=e1.loss_grad(x,y,a,b)
        for w,g in [(a,ga),(b,gb)]:
            for idx in [(0,0),(1,2),(3,1)]:
                original=w[idx];w[idx]=original+1e-6;hi=e1.loss_grad(x,y,a,b)[0]
                w[idx]=original-1e-6;lo=e1.loss_grad(x,y,a,b)[0];w[idx]=original
                self.assertAlmostEqual(g[idx],(hi-lo)/2e-6,places=6)

    def test_empty_unknown_text_does_not_create_features(self):
        x=e1.text_matrix(['xyz'],['amber'],np.ones(1))
        self.assertTrue(np.isfinite(x).all());self.assertEqual(float(x.sum()),0)

    def test_unknown_material_does_not_renormalize_known_mass(self):
        x=e1.formula_matrix([{'components':[{'ingredient_id':'a','weight_percent':25},
             {'ingredient_id':'unknown','weight_percent':75}]}],['a'])
        self.assertEqual(float(x[0,0]),.5)

if __name__=='__main__':unittest.main()
