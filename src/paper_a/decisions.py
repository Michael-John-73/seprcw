"""Require operational grounds and pilot evidence before confirmatory design lock."""
from pathlib import Path
from .io import read_json,root,sha,assert_no_final_outcome_access
KEYS=('margin_embed','margin_decode','margin_joint','min_psnr','min_ssim',
      'max_decode_seconds','quality_gap_db','test_n','external_test_n')

def validate_decisions(cfg):
    assert_no_final_outcome_access(cfg)
    record=read_json(root(cfg)/'decisions.json')
    if record.get('study_id')!=cfg['study']['id'] or record.get('protocol_version')!=cfg['study']['protocol_version']:
        raise RuntimeError('decision record study/version mismatch')
    for k in ['decided_by','decided_at']:
        if not isinstance(record.get(k),str) or not record[k].strip():
            raise RuntimeError('decision record requires '+k)
    requirements=record.get('requirements',{})
    if set(requirements)!=set(KEYS):raise RuntimeError('decision requirements missing/unknown')
    for k in KEYS:
        row=requirements[k]
        if cfg['evaluation'][k] is None or row.get('value')!=cfg['evaluation'][k]:
            raise RuntimeError('unresolved or mismatched decision value: '+k)
        reason=row.get('rationale')
        if not isinstance(reason,str) or not reason.strip() or reason.strip().lower() in ['todo','pending','tbd']:
            raise RuntimeError('decision rationale missing: '+k)
        if k.startswith('margin_') and row.get('basis')!='operational_requirement':
            raise RuntimeError('margin must derive from operational requirement, not observed pilot gain')
    evidence=record.get('evidence',[])
    if not {'pilot','operational_requirement'}.issubset({x.get('role') for x in evidence}):
        raise RuntimeError('decision needs pilot and operational-requirement evidence')
    paths=[]
    for item in evidence:
        p=Path(item['path'])
        if not p.is_file() or sha(p)!=item['sha256']:
            raise RuntimeError('decision evidence missing/changed')
        paths.append(p)
    return record,paths
