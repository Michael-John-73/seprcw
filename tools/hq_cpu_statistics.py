"""One H-Q contrast, paired by source and stratified by photographic/AI domain."""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import norm

AREAS = (0.8, 0.5)
RULES = ('E1', 'E2')

def plan_n(halfwidth=.03, variance=.1982323232323232):
    if not 0 < halfwidth < 1 or not math.isfinite(variance) or variance <= 0:
        raise ValueError('invalid precision planning inputs')
    n = 2 * math.ceil(norm.ppf(.975)**2 * variance / halfwidth**2 / 2)
    return {'n_sources': n, 'per_domain': n//2, 'halfwidth': halfwidth,
            'planning_variance': variance, 'precision_guaranteed': False}

def paired_values(rows, cohort):
    ids = {s['source_id']: s['domain'] for s in cohort}
    if len(ids) != len(cohort) or set(ids.values()) != {'ai', 'real'}:
        raise ValueError('invalid cohort identifiers/domains')
    lookup = {}
    for row in rows:
        key = row['source_id'], row['rule'], row['area']
        if key in lookup or row['source_id'] not in ids:
            raise ValueError('duplicate/unregistered result')
        if row['domain'] != ids[row['source_id']] or row['ER'] not in (0, 1):
            raise ValueError('invalid result domain or ER')
        lookup[key] = row
    expected = {(sid, rule, area) for sid in ids for rule in RULES for area in AREAS}
    if set(lookup) != expected:
        raise ValueError('incomplete or extra result panel')
    result = []
    for sid in sorted(ids):
        for area in AREAS:
            a, b = lookup[sid,'E1',area], lookup[sid,'E2',area]
            if any(a[k] != b[k] for k in ('message','top','left','height','width')):
                raise ValueError('unpaired message/crop')
        e1 = np.mean([lookup[sid,'E1',a]['ER'] for a in AREAS])
        e2 = np.mean([lookup[sid,'E2',a]['ER'] for a in AREAS])
        result.append({'source_id':sid,'domain':ids[sid], 'E1':float(e1),'E2':float(e2),
                       'difference':float(e2-e1)})
    return result

def stratified_interval(values, reps=10000, seed=20261026, chunk=256):
    if type(reps) is not int or reps < 1 or type(chunk) is not int or chunk < 1:
        raise ValueError('invalid bootstrap budget')
    arrays = [np.array([v['difference'] for v in values if v['domain']==domain], dtype=np.float64)
              for domain in ('ai','real')]
    if any(len(a)<2 or not np.isfinite(a).all() or np.any(np.abs(a)>1) for a in arrays):
        raise ValueError('need finite bounded differences in both domains')
    if len(arrays[0]) != len(arrays[1]):
        raise ValueError('registered balanced allocation required')
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    for start in range(0,reps,chunk):
        count = min(chunk,reps-start)
        draws[start:start+count] = sum(.5*a[rng.integers(len(a),size=(count,len(a)))].mean(1)
                                     for a in arrays)
    mean = sum(.5*float(a.mean()) for a in arrays)
    if np.ptp(draws) == 0:
        # Known response range [-1,1]; no zero population uncertainty claim.
        radius = math.sqrt(2*math.log(2/.05)/sum(map(len,arrays)))
        low, high = max(-1.,mean-radius), min(1.,mean+radius)
        method = 'bounded_Hoeffding_fallback_for_degenerate_empirical_draws'
    else:
        low, high = map(float,np.quantile(draws,[.025,.975]))
        method = 'stratified_source_paired_percentile_bootstrap'
    return {'difference':mean, 'low':low,'high':high,'confidence':.95,'draws':reps,
            'seed':seed,'method':method,'domain_weights':{'ai':.5,'real':.5},
            'n_sources':sum(map(len,arrays)), 'exact_finite_sample_bootstrap_coverage_claimed':False}

def quality_summary(rows, cohort):
    ids = {s['source_id'] for s in cohort}
    lookup = {(r['source_id'],r['rule']):r for r in rows}
    if len(lookup) != len(rows) or set(lookup) != {(sid,r) for sid in ids for r in RULES}:
        raise ValueError('quality panel incomplete/duplicated')
    result = {}
    for rule in RULES:
        records = [lookup[sid,rule] for sid in sorted(ids)]
        for key in ('psnr','ssim'):
            a = np.asarray([r[key] for r in records],dtype=float)
            if not np.isfinite(a).all() or any(r.get('psnr_infinite',False) for r in records):
                raise ValueError('nonfinite/infinite quality; no implicit success')
            result[f'{rule}_{key}_mean'] = float(a.mean())
    for key in ('psnr','ssim'):
        a = np.array([lookup[sid,'E2'][key]-lookup[sid,'E1'][key] for sid in sorted(ids)])
        result[key+'_paired_difference'] = {'mean':float(a.mean()), 'min':float(a.min()),
            'q05':float(np.quantile(a,.05)), 'median':float(np.median(a)),
            'q95':float(np.quantile(a,.95)), 'max':float(a.max())}
    result['psnr_difference_abs'] = abs(result['E2_psnr_mean']-result['E1_psnr_mean'])
    result['ssim_difference_abs'] = abs(result['E2_ssim_mean']-result['E1_ssim_mean'])
    result['matched'] = result['psnr_difference_abs'] <= .25 and result['ssim_difference_abs'] <= .005
    result['criteria'] = {'mean_PSNR_absolute_difference_dB':.25,'mean_SSIM_absolute_difference':.005}
    return result

def hq_verdict(interval, quality_matched):
    if not quality_matched:
        return 'QUALITY_MATCH_NOT_CONFIRMED'
    if interval['low'] > 0:
        return 'SUPPORTED_AT_REGISTERED_QUALITY_POINT'
    if interval['high'] <= 0:
        return 'POSITIVE_GAIN_NOT_SUPPORTED'
    return 'INCONCLUSIVE'
