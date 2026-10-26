import copy
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from hq_cpu_statistics import plan_n, paired_values, stratified_interval, quality_summary, hq_verdict
from hq_independent_cpu import NearIndex

def fixture():
    cohort=[{'source_id':str(i),'domain':'ai' if i<2 else 'real'} for i in range(4)]
    rows=[];quality=[]
    for s in cohort:
        for rule in ('E1','E2'):
            quality.append({**s,'rule':rule,'psnr':38.+(.1 if rule=='E2' else 0),
                            'ssim':.97+(.001 if rule=='E2' else 0),'psnr_infinite':False})
            for area in (.8,.5):
                rows.append({**s,'rule':rule,'area':area,'message':'123456789ABC',
                             'top':3,'left':7,'height':144,'width':144,
                             'ER':int(rule=='E2')})
    return cohort,rows,quality

class HqCpuTests(unittest.TestCase):
    def test_authorized_precision_sources_are_not_crop_rows(self):
        self.assertEqual(plan_n(.03)['n_sources'],848)
        self.assertEqual(plan_n(.03)['per_domain'],424)
        self.assertFalse(plan_n()['precision_guaranteed'])
    def test_pairing_returns_sources_not_four_case_rows(self):
        cohort,rows,_=fixture();values=paired_values(rows,cohort)
        self.assertEqual(len(values),4)
        self.assertEqual([v['difference'] for v in values],[1.]*4)
    def test_incomplete_or_duplicate_panel_rejected(self):
        cohort,rows,_=fixture()
        with self.assertRaises(ValueError):paired_values(rows[:-1],cohort)
        with self.assertRaises(ValueError):paired_values(rows+[copy.deepcopy(rows[0])],cohort)
    def test_unpaired_crop_or_message_rejected(self):
        cohort,rows,_=fixture();rows[0]['left']+=1
        with self.assertRaises(ValueError):paired_values(rows,cohort)
    def test_stratified_constant_domains_do_not_claim_zero_uncertainty(self):
        values=[{'domain':d,'difference':v} for d,v in [('ai',1.),('real',0.)] for _ in range(50)]
        result=stratified_interval(values,reps=100,seed=3)
        self.assertEqual(result['difference'],.5)
        self.assertLess(result['low'],.5);self.assertGreater(result['high'],.5)
        self.assertIn('fallback',result['method'])
    def test_bootstrap_seed_and_chunk_do_not_change_estimand(self):
        values=[{'domain':d,'difference':v} for d in ('ai','real') for v in [0.,.5,1.]*20]
        a=stratified_interval(values,500,7,50);b=stratified_interval(values,500,7,50)
        self.assertEqual(a,b);self.assertEqual(a['difference'],.5)
        self.assertEqual(a['confidence'],.95)
    def test_unbalanced_or_invalid_source_differences_rejected(self):
        with self.assertRaises(ValueError):stratified_interval([{'domain':'ai','difference':0.}]*2)
        with self.assertRaises(ValueError):stratified_interval([{'domain':d,'difference':float('nan')} for d in ('ai','real') for _ in range(2)])
    def test_quality_gate_uses_both_metrics(self):
        cohort,_,rows=fixture();self.assertTrue(quality_summary(rows,cohort)['matched'])
        for r in rows:
            if r['rule']=='E2':r['ssim']-=.02
        self.assertFalse(quality_summary(rows,cohort)['matched'])
    def test_infinite_quality_cannot_be_implicit_pass(self):
        cohort,_,rows=fixture();rows[0]['psnr_infinite']=True
        with self.assertRaises(ValueError):quality_summary(rows,cohort)
    def test_zero_null_is_valid_for_new_HQ_without_rewriting_old_margins(self):
        self.assertEqual(hq_verdict({'low':.1,'high':.2},True),'SUPPORTED_AT_REGISTERED_QUALITY_POINT')
        self.assertEqual(hq_verdict({'low':-.1,'high':0.},True),'POSITIVE_GAIN_NOT_SUPPORTED')
        self.assertEqual(hq_verdict({'low':-.1,'high':.1},True),'INCONCLUSIVE')
        self.assertEqual(hq_verdict({'low':.1,'high':.2},False),'QUALITY_MATCH_NOT_CONFIRMED')
    def test_complete_hamming_index_for_four_differing_bits(self):
        index=NearIndex();index.add(0,'old')
        for positions in [(0,13,26,39),(12,25,38,51),(1,15,30,63)]:
            self.assertEqual(index.match(sum(1<<p for p in positions)),'old')
        self.assertIsNone(index.match(sum(1<<p for p in (0,13,26,39,52))))

if __name__=='__main__':unittest.main(verbosity=2)
