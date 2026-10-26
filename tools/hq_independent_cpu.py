"""CPU-only, frozen H-Q confirmation. Existing S/E pipelines remain unchanged.

Commands: verify-old, admit, lock, run, recount, analyze. No training, acceptance
calibration, GPU fallback, strength selection, or outcome-dependent exclusions.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import sys
import tarfile
import time
import traceback
import zipfile
import cv2
import numpy as np
from tqdm import tqdm
from hq_cpu_statistics import plan_n, paired_values, stratified_interval, quality_summary, hq_verdict

ARCHIVE_SHA = 'cc3b691e4e84f86672ef7984d06350ee9dbc17dafd1a7f98dba5caa45ed5e786'
ARCHIVE_URL = 'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/527749109c86eda3c82027143e45964e75cbf772/images/part-000002.zip'
FIXED = {'E1_delta':36,'E2_delta':16,'eta':2.,'decoder':'D0','areas':[.8,.5],
         'payload_bits':48,'pilots':16,'tile_side':8,'pattern_seed':20261008,
         'crop_seed':20261021,'sample_seed':20261026,'n_sources':848,'per_domain':424,
         'target_halfwidth':.03,'bootstrap_reps':10000,'bootstrap_seed':20261026,
         'PSNR_tolerance':.25,'SSIM_tolerance':.005,'dhash_distance':4}

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def stamp():
    return datetime.now(timezone.utc).isoformat()

def event(message,**fields):
    print(json.dumps({'at':stamp(),'event':message,**fields},ensure_ascii=False,allow_nan=False),flush=True)

def engine(project):
    sys.path.insert(0,str(Path(project)/'src'))
    import torch
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    from paper_a import codec
    from paper_a import data
    from paper_a.compute import cuda_enabled
    if cuda_enabled():
        raise RuntimeError('CPU confirmation forbids active CUDA dispatch')
    return codec,data

def engine_hashes(project):
    base=Path(project)/'src'/'paper_a'
    return {p.name:sha(p) for p in sorted(base.glob('*.py'))}

def runner_hashes():
    return {Path(__file__).name:sha(__file__),
            'hq_cpu_statistics.py':sha(Path(__file__).with_name('hq_cpu_statistics.py'))}

def config():
    return {'study':{'seed':FIXED['crop_seed']},'data':{'max_side':512,'min_side':144},
            'evaluation':{'areas':FIXED['areas']}}

def prepare_pixels(path,enforce_min=True):
    """Legacy short-side cubic preparation, PNG lossless, then max-side area resize."""
    image=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if image is None: raise ValueError('unreadable image')
    h,w=image.shape[:2]
    if min(h,w)<256:
        scale=256/min(h,w)
        image=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_CUBIC)
    # PNG roundtrip is lossless; retain identical integer pixels before max-side resize.
    h,w=image.shape[:2]; scale=min(1.,512/max(h,w))
    if scale<1:
        image=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
    if enforce_min and min(image.shape[:2])<144: raise ValueError('post-preparation minimum side below 144')
    return image

def pixel_hash(image):
    return hashlib.sha256(image.tobytes()+str(image.shape).encode()).hexdigest()

def image_dhash(image):
    tiny=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(9,8),interpolation=cv2.INTER_AREA)
    return int(''.join(map(str,(tiny[:,1:]>tiny[:,:-1]).ravel().astype(int))),2)

def cpu_environment():
    import torch, pywt, skimage, scipy
    return {'python':platform.python_version(),'processor':platform.processor(),
            'logical_cpu_count':os.cpu_count(),'numpy':np.__version__,'opencv':cv2.__version__,
            'torch':torch.__version__,'pywavelets':pywt.__version__,'scipy':scipy.__version__,
            'skimage':skimage.__version__,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],
            'mathematical_backend':'CPU NumPy/PyWavelets/OpenCV; no CUDA dispatch',
            'inference_threads_per_worker':1,'gpu_used':False}

def write_png(path,image):
    ok,data=cv2.imencode('.png',image)
    if not ok: raise RuntimeError('PNG encode failed')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_bytes(data.tobytes())

class NearIndex:
    """Complete Hamming<=4 candidate retrieval using five non-overlapping bands."""
    def __init__(self): self.buckets={}; self.values=[]
    def keys(self,value):
        start=0
        for band,length in enumerate((13,13,13,13,12)):
            yield band,(value>>start)&((1<<length)-1)
            start+=length
    def match(self,value):
        candidates=set()
        for key in self.keys(value): candidates.update(self.buckets.get(key,()))
        for i in sorted(candidates):
            if (value^self.values[i][0]).bit_count()<=4: return self.values[i][1]
        return None
    def add(self,value,identity):
        n=len(self.values); self.values.append((value,identity))
        for key in self.keys(value): self.buckets.setdefault(key,[]).append(n)

def raw_paths(legacy):
    legacy=Path(legacy)
    return {'real':legacy/'raw'/'coco_val2017'/'val2017','ai':legacy/'raw'/'diffusiondb_part1'}

def old_indices(legacy):
    return {d:read_csv(Path(legacy)/'work'/f'index_{d}.csv') for d in ('real','ai')}

def archive_jsons(path,names):
    with tarfile.open(path,'r:gz') as t:
        members={m.name.removeprefix('./'):m for m in t.getmembers() if m.isfile()}
        return {n:json.load(t.extractfile(members[n])) for n in names}

def verify_old(args):
    """Twenty fixed old validation sources, balanced; compare recorded CUDA outputs."""
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    target=out/'cpu_reference_equivalence.json'
    if target.exists(): raise FileExistsError('preserve existing preflight attempt')
    codec,data=engine(args.project)
    old=archive_jsons(args.reference_archive,['input_manifest.json','selected_pair_validation_rows.json',
                                             'quality_E1_36.json','quality_E2_16.json'])
    source_rows=old['input_manifest.json']['validation']
    chosen=sum([sorted([s for s in source_rows if s['domain']==d],key=lambda s:s['source_id'])[:10]
                for d in ('ai','real')],[])
    indices=old_indices(args.legacy_data)
    lookup={d:{r['source_file']:r for r in indices[d]} for d in indices}
    ref={(r['source_id'],r['rule'],r['area']):r for r in old['selected_pair_validation_rows.json']}
    quality_ref={(r['source_id'],rule):r for rule,name in [('E1','quality_E1_36.json'),('E2','quality_E2_16.json')]
                 for r in old[name]}
    mismatches=[]; checked=[]; start=time.perf_counter()
    for source in tqdm(chosen,desc='CPU versus stored CUDA: old sources',file=sys.stdout):
        d=source['domain']; index=lookup[d][source['original_source_file']]
        image=prepare_pixels(Path(args.legacy_data)/'work'/d/index['work_file'])
        if pixel_hash(image)!=source['pixel_sha256']:
            mismatches.append({'source_id':source['source_id'],'kind':'prepared_pixels'})
            continue
        word=ref[source['source_id'],'E1',.8]['message']
        bits=np.unpackbits(np.frombuffer(bytes.fromhex(word),dtype=np.uint8))
        for rule,delta in [('E1',36),('E2',16)]:
            cc=replace(codec.CodecConfig(),delta=delta)
            marked,_=codec.png_roundtrip(codec.embed(image,bits,cc,rule,2.))
            q=data.quality(image,marked); expected_q=quality_ref[source['source_id'],rule]
            for metric in ('psnr','ssim'):
                if abs(q[metric]-expected_q[metric])>1e-8:
                    mismatches.append({'source_id':source['source_id'],'rule':rule,'kind':metric,
                                       'cpu':q[metric],'stored_cuda':expected_q[metric]})
            for info,crop in data.crop_panel(marked,config(),source['source_id']):
                bank=codec.collect(crop,cc,'search'); pred,_=codec.baseline(bank,'D0')
                expected=ref[source['source_id'],rule,info['area']]
                same_bits=np.array_equal(pred,expected['predicted_bits'])
                same_phase=list(map(int,bank['offset']))==expected['phase']
                if not same_bits or not same_phase:
                    mismatches.append({'source_id':source['source_id'],'rule':rule,'area':info['area'],
                                       'kind':'bits_or_phase','same_bits':same_bits,'same_phase':same_phase})
                checked.append({'source_id':source['source_id'],'rule':rule,'area':info['area'],
                                'same_bits':same_bits,'same_phase':same_phase})
    result={'status':'PASS' if not mismatches and len(checked)==80 else 'FAIL',
            'reference':'stored original CUDA results; no GPU used in this check','old_sources':20,
            'cases':len(checked),'checks':checked,'mismatches':mismatches,
            'wall_s':time.perf_counter()-start,'engine_hashes':engine_hashes(args.project),
            'reference_archive_sha256':sha(args.reference_archive),'at':stamp()}
    save(target,result); event('CPU_REFERENCE_PREFLIGHT',status=result['status'],mismatches=len(mismatches))
    if result['status']!='PASS': raise RuntimeError('CPU/reference numerical mismatch; new cohort must remain unopened')

def extract_ai(archive,destination):
    if sha(archive)!=ARCHIVE_SHA: raise ValueError('official DiffusionDB archive hash mismatch')
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    seen=set()
    with zipfile.ZipFile(archive) as z:
        for member in tqdm(z.infolist(),desc='Extract fresh DiffusionDB shard',file=sys.stdout):
            if member.is_dir(): continue
            name=PurePosixPath(member.filename).name
            if Path(name).suffix.lower() not in ('.png','.json'): continue
            if name in seen: raise ValueError('duplicate archive basename')
            seen.add(name)
            path=destination/name
            content=z.read(member)
            if path.exists() and path.read_bytes()!=content: raise RuntimeError('existing extracted file changed')
            if not path.exists(): path.write_bytes(content)
    images=list(destination.glob('*.png'))
    if len(images)!=1000: raise RuntimeError('expected 1000 images in new shard')
    return images

def admit(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'design.json').exists(): raise FileExistsError('preserve prior design/admission attempt')
    preflight=load(out/'cpu_reference_equivalence.json')
    if preflight['status']!='PASS' or preflight['engine_hashes']!=engine_hashes(args.project):
        raise RuntimeError('current CPU engine must pass old-source reference comparison')
    if plan_n()['n_sources']!=848: raise RuntimeError('precision calculation changed')
    selection=load(args.selection)
    if (selection['selected']['delta_E1'],selection['selected']['delta_E2'])!=(36,16):
        raise ValueError('selected pair differs from authorized follow-up')
    dataset=Path(args.dataset); dataset.mkdir(parents=True,exist_ok=True)
    design={'id':'H_Q_CPU_INDEPENDENT_20261026','at':stamp(),'backend':'cpu_only_explicit_confirmation',
            'user_precision_decision':'95% CI planning halfwidth +/-3pp, 848 sources, 424 per domain',
            'fixed':FIXED,'planning':plan_n(),'selection_path':str(Path(args.selection).resolve()),
            'selection_sha256':sha(args.selection),'population':{
                'real':'COCO val2017, excluding all previously prepared sources',
                'ai':'DiffusionDB 2M fresh shard part-000002; old shard part-000001 fully excluded',
                'claim':'same-dataset fresh-source evaluation; not same original finite shard or unseen-generator generalization',
                'archive_url':ARCHIVE_URL,'archive_sha256':ARCHIVE_SHA},
            'preparation':{'short_side_below_256':'cubic upscale','then':'lossless PNG integer pixels',
                           'max_side':512,'max_side_interpolation':'INTER_AREA','min_post_side':144},
            'inference':'48bit D0, blind search; no reference/crop coordinates passed to decoder',
            'new_training':False,'new_acceptance_calibration':False,
            'missing_case_policy':'coverage erasure=ER0; unexpected numerical/I/O errors invalidate run; no silent replacement',
            'quality_gate':'abs mean PSNR difference<=.25 and abs mean SSIM difference<=.005',
            'analysis':'single contrast, 10000 domain-stratified source-paired draws; weights .5/.5; nominal two-sided 95%',
            'degenerate_interval':'bounded Hoeffding fallback on [-1,1], alpha=.05',
            'data_access':'input-only preparation/duplicate audit before lock; no new embedding/recovery outcomes',
            'engine_hashes':engine_hashes(args.project),'runner_hashes':runner_hashes(),
            'environment':cpu_environment()}
    save(out/'design.json',design)
    codec,data=engine(args.project)
    indices=old_indices(args.legacy_data); raws=raw_paths(args.legacy_data)
    historical=[]; raw_hashes=set(); pixels=set(); near=NearIndex()
    for domain in ('real','ai'):
        for row in tqdm(indices[domain],desc=f'Historical exclusion {domain}',file=sys.stdout):
            raw=raws[domain]/row['source_file']; working=Path(args.legacy_data)/'work'/domain/row['work_file']
            image=prepare_pixels(working,enforce_min=False)
            entry={'domain':domain,'original_source_file':row['source_file'], 'raw_sha256':sha(raw),
                   'prepared_pixel_sha256':pixel_hash(image),'dhash':image_dhash(image)}
            historical.append(entry); raw_hashes.add(entry['raw_sha256']);pixels.add(entry['prepared_pixel_sha256'])
            near.add(entry['dhash'],'old:'+domain+':'+row['source_file'])
    if len(historical)!=2000: raise RuntimeError('expected conservative exclusion of all prior working images')
    save(out/'historical_exclusion.json',{'history_inputs':{
        d:{'index':str(Path(args.legacy_data)/'work'/f'index_{d}.csv'),
           'sha256':sha(Path(args.legacy_data)/'work'/f'index_{d}.csv')} for d in indices},'sources':historical})
    fresh_ai=extract_ai(Path(args.dataset)/'raw'/'diffusiondb_part-000002.zip',Path(args.dataset)/'raw'/'diffusiondb_part2')
    old_names={d:{r['source_file'] for r in indices[d]} for d in indices}
    pools={'real':[p for p in raws['real'].glob('*.jpg') if p.name not in old_names['real']],
           'ai':[p for p in fresh_ai if p.name not in old_names['ai']]}
    selected=[]; exclusions=[]; input_counts={d:len(p) for d,p in pools.items()}
    for domain in ('real','ai'):
        ranked=sorted(pools[domain],key=lambda p:hashlib.sha256(f"H-Q|20261026|{domain}|{p.name}".encode()).hexdigest())
        admitted=0
        for path in tqdm(ranked,desc=f'Fresh source admission {domain}',file=sys.stdout):
            if admitted==424: break
            try:
                image=prepare_pixels(path)
            except ValueError as exc:
                exclusions.append({'domain':domain,'path':str(path),'reason':str(exc)});continue
            raw_hash=sha(path); ph=pixel_hash(image); dh=image_dhash(image)
            match=near.match(dh)
            reason='duplicate_raw_bytes' if raw_hash in raw_hashes else ('duplicate_processed_pixels' if ph in pixels else ('dhash_near_duplicate' if match else None))
            if reason:
                exclusions.append({'domain':domain,'path':str(path),'reason':reason,'near_match':match});continue
            sid=hashlib.sha256((domain+':'+path.name).encode()).hexdigest()[:24]
            prepared=dataset/'prepared'/f'{sid}.png';write_png(prepared,image)
            selected.append({'source_id':sid,'domain':domain,'dataset':'COCO val2017' if domain=='real' else 'DiffusionDB 2M part-000002',
                             'original_source_file':path.name,'raw_path':str(path.resolve()),'raw_sha256':raw_hash,
                             'path':str(prepared.resolve()),'sha256':sha(prepared),'pixel_sha256':ph,
                             'height':image.shape[0],'width':image.shape[1],'dhash':dh,'previous_outcomes_used':False})
            raw_hashes.add(raw_hash);pixels.add(ph);near.add(dh,'new:'+sid);admitted+=1
        if admitted!=424:
            save(out/'admission_failure.json',{'domain':domain,'admitted':admitted,'requested':424,'exclusions':exclusions})
            raise RuntimeError('insufficient audited fresh sources; no scientific outcomes opened')
    selected=sorted(selected,key=lambda r:r['source_id'])
    if len({r['source_id'] for r in selected})!=848: raise RuntimeError('source collision')
    save(out/'cohort.json',selected);save(out/'input_exclusions.json',exclusions)
    # New role-specific random words; no reference enters blind decoding.
    rng=np.random.default_rng(20261026);pool=[];used={int('B3EC907BB19E',16)}
    old=archive_jsons(args.reference_archive,['selected_pair_validation_rows.json','diagnostic_B_rows.json'])
    used.update(int(r['message'],16) for rows in old.values() for r in rows)
    while len(pool)<1024:
        word=int(rng.integers(0,1<<48))
        if word not in used: used.add(word);pool.append(f'{word:012X}')
    messages={r['source_id']:pool[int.from_bytes(hashlib.sha256((r['source_id']+'|assignment').encode()).digest()[:8],'big')%len(pool)] for r in selected}
    save(out/'messages.json',{'pool':pool,'assignments':messages,'role':'hq_independent','seed':20261026})
    save(out/'admission.json',{'status':'INPUT_ONLY_AUDIT_PASS','n_sources':848,'domains':dict(Counter(r['domain'] for r in selected)),
        'historical_sources_excluded':2000,'prior_original_file_names_overlap':0,'raw_bytes_overlap':0,'processed_pixels_overlap':0,
        'dhash_within_distance4_accepted':0,'dhash_is_heuristic':True,'input_pool_counts':input_counts,'exclusions':len(exclusions),
        'scope':'same-dataset fresh shard; neither original finite-shard resampling nor generator-generalization claim',
        'outcomes_opened':False,'at':stamp()})
    event('ADMISSION_COMPLETE',sources=848,exclusions=len(exclusions))

def freeze(args):
    out=Path(args.out)
    if (out/'lock.json').exists() or (out/'opened.json').exists(): raise FileExistsError('design already frozen/opened')
    design=load(out/'design.json');cohort=load(out/'cohort.json')
    if design['fixed']!=FIXED or design['engine_hashes']!=engine_hashes(args.project) or design['runner_hashes']!=runner_hashes():
        raise RuntimeError('design/code changed after admission')
    if Counter(r['domain'] for r in cohort)!={'real':424,'ai':424}: raise ValueError('cohort allocation')
    for row in tqdm(cohort,desc='CPU experiment pre-open file hashes',file=sys.stdout):
        if sha(row['path'])!=row['sha256']: raise RuntimeError('prepared file hash changed')
    names=['design.json','cohort.json','messages.json','admission.json','historical_exclusion.json','input_exclusions.json','cpu_reference_equivalence.json']
    save(out/'lock.json',{'at':stamp(),'status':'FROZEN_BEFORE_NEW_OUTCOMES','files':{n:sha(out/n) for n in names},
                          'engine_hashes':engine_hashes(args.project),'runner_hashes':runner_hashes()})
    event('LOCKED_BEFORE_NEW_OUTCOMES',sources=848)

def verify_lock(args):
    out=Path(args.out);lock=load(out/'lock.json')
    if lock['engine_hashes']!=engine_hashes(args.project) or lock['runner_hashes']!=runner_hashes():
        raise RuntimeError('locked CPU engine/runner modified')
    for name,value in lock['files'].items():
        if sha(out/name)!=value: raise RuntimeError('locked artifact changed: '+name)
    return lock

def task_worker(task):
    project,out,row,word=task
    codec,data=engine(project)
    if sha(row['path'])!=row['sha256']: raise RuntimeError('source changed at inference')
    image=cv2.imread(row['path'],cv2.IMREAD_COLOR)
    if image is None or pixel_hash(image)!=row['pixel_sha256']: raise RuntimeError('invalid prepared pixels')
    bits=np.unpackbits(np.frombuffer(bytes.fromhex(word),dtype=np.uint8))
    rows=[];qualities=[];start=time.perf_counter()
    for rule,delta in [('E1',36),('E2',16)]:
        cc=replace(codec.CodecConfig(),delta=delta)
        marked,png=codec.png_roundtrip(codec.embed(image,bits,cc,rule,2.))
        stored=Path(out)/'marked'/f"{row['source_id']}_{rule}.png";stored.write_bytes(png)
        qualities.append({'source_id':row['source_id'],'domain':row['domain'],'rule':rule,'delta':delta,
                          'message':word,'marked_path':str(stored.resolve()),'marked_sha256':sha(stored),**data.quality(image,marked)})
        for info,crop in data.crop_panel(marked,config(),row['source_id']):
            before=time.perf_counter()
            bank=codec.collect(crop,cc,'search') # NEVER supplies origin or reference.
            pred,_=codec.baseline(bank,'D0')
            valid=bool(np.all(bank['counts']>0));exact=int(valid and np.array_equal(pred,bits))
            rows.append({'source_id':row['source_id'],'domain':row['domain'],'rule':rule,'delta':delta,
                         'message':word,**info,'ER':exact,'BER':float(np.mean(pred!=bits)),
                         'coverage_valid':valid,'predicted_bits':pred.tolist(),'phase':list(map(int,bank['offset'])),
                         'phase_matches_origin':tuple(bank['offset'])==(info['top']%64,info['left']%64),
                         'decode_s':time.perf_counter()-before,'backend':'cpu'})
    return rows,qualities,time.perf_counter()-start

def run(args):
    out=Path(args.out);verify_lock(args)
    opened=out/'opened.json'
    with opened.open('x',encoding='utf-8') as f:
        json.dump({'status':'OUTCOMES_OPENED','at':stamp(),'lock_sha256':sha(out/'lock.json'),
                   'backend':'cpu','workers':args.workers,'no_GPU_fallback':True},f,indent=2)
    (out/'marked').mkdir()
    cohort=load(out/'cohort.json');messages=load(out/'messages.json')['assignments']
    tasks=[(str(Path(args.project).resolve()),str(out.resolve()),r,messages[r['source_id']]) for r in cohort]
    all_rows=[];all_quality=[];start=time.perf_counter()
    try:
        with (out/'rows.jsonl').open('x',encoding='utf-8') as rf,(out/'quality.jsonl').open('x',encoding='utf-8') as qf:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures={executor.submit(task_worker,t):t[2]['source_id'] for t in tasks}
                progress=tqdm(as_completed(futures),total=len(futures),desc='CPU H-Q independent sources',file=sys.stdout)
                for future in progress:
                    rows,quality,elapsed=future.result()
                    for row in rows: rf.write(json.dumps(row,allow_nan=False)+'\n')
                    for row in quality: qf.write(json.dumps(row,allow_nan=False)+'\n')
                    rf.flush();qf.flush();all_rows.extend(rows);all_quality.extend(quality)
                    done=len(all_rows)//4
                    if done%16==0 or done==848:
                        save(out/'status.json',{'phase':'CPU_INFERENCE','completed_sources':done,'total_sources':848,
                             'completed_rows':len(all_rows),'elapsed_s':time.perf_counter()-start,'gpu_used':False})
                        event('CPU_PROGRESS',completed=done,total=848,rows=len(all_rows))
        paired_values(all_rows,cohort);quality_summary(all_quality,cohort)
        save(out/'rows.json',sorted(all_rows,key=lambda r:(r['source_id'],r['rule'],-r['area'])))
        save(out/'quality.json',sorted(all_quality,key=lambda r:(r['source_id'],r['rule'])))
        save(out/'prediction_lock.json',{'at':stamp(),'rows_sha256':sha(out/'rows.json'),
             'quality_sha256':sha(out/'quality.json'),'sources':848,'rows':len(all_rows),'quality_rows':len(all_quality),
             'wall_s':time.perf_counter()-start,'workers':args.workers,'gpu_used':False})
        save(out/'status.json',{'phase':'CPU_PREDICTIONS_LOCKED','completed_sources':848,'rows':len(all_rows),'gpu_used':False})
        event('CPU_PREDICTIONS_LOCKED',sources=848,rows=len(all_rows),wall_s=time.perf_counter()-start)
    except Exception:
        save(out/'execution_failure.json',{'at':stamp(),'status':'INVALID_EXECUTION','traceback':traceback.format_exc(),
                                         'completed_sources':len(all_rows)//4,'rows':len(all_rows),'no_source_replacement':True})
        raise

def verify_predictions(args):
    verify_lock(args);out=Path(args.out)
    p=load(out/'prediction_lock.json')
    if sha(out/'rows.json')!=p['rows_sha256'] or sha(out/'quality.json')!=p['quality_sha256']:
        raise RuntimeError('locked predictions modified')
    return load(out/'rows.json'),load(out/'quality.json'),load(out/'cohort.json')

def recount(args):
    rows,qualities,cohort=verify_predictions(args);out=Path(args.out)
    for row in tqdm(rows,desc='CPU independent bit/ER recount',file=sys.stdout):
        bits=np.unpackbits(np.frombuffer(bytes.fromhex(row['message']),dtype=np.uint8))
        pred=np.asarray(row['predicted_bits'])
        if pred.shape!=(48,) or not np.isin(pred,[-1,0,1]).all():raise RuntimeError('invalid predicted bits')
        exact=int(row['coverage_valid'] and np.array_equal(pred,bits))
        if row['ER']!=exact or abs(row['BER']-np.mean(pred!=bits))>1e-12:raise RuntimeError('ER/BER recount mismatch')
    byid={r['source_id']:r for r in cohort}
    codec,data=engine(args.project)
    for row in tqdm(qualities,desc='CPU stored PNG/quality recount',file=sys.stdout):
        if sha(row['marked_path'])!=row['marked_sha256']:raise RuntimeError('marked PNG changed')
        original=cv2.imread(byid[row['source_id']]['path'],cv2.IMREAD_COLOR)
        marked=cv2.imread(row['marked_path'],cv2.IMREAD_COLOR)
        q=data.quality(original,marked)
        if any(abs(q[k]-row[k])>1e-10 for k in ('psnr','ssim')):raise RuntimeError('quality recount mismatch')
    values=paired_values(rows,cohort);q=quality_summary(qualities,cohort)
    save(out/'recount.json',{'status':'PASS','at':stamp(),'rows':len(rows),'sources':len(values),
                            'quality_rows':len(qualities),'quality_matched':q['matched'],
                            'prediction_lock_sha256':sha(out/'prediction_lock.json'),'gpu_used':False})
    event('RECOUNT_PASS',sources=len(values),rows=len(rows))

def analyze(args):
    rows,qualities,cohort=verify_predictions(args);out=Path(args.out)
    rec=load(out/'recount.json')
    if rec['status']!='PASS' or rec['prediction_lock_sha256']!=sha(out/'prediction_lock.json'):
        raise RuntimeError('current independent recount required')
    values=paired_values(rows,cohort);q=quality_summary(qualities,cohort)
    ci=stratified_interval(values,10000,20261026,256)
    summary={'hypothesis':'H-Q','backend':'cpu','gpu_used':False,'sources':len(cohort),
             'source_allocation':dict(Counter(s['domain'] for s in cohort)),
             'quality':q,'effect':ci,'verdict':hq_verdict(ci,q['matched']),
             'primary_recovery':{rule:{'exact':sum(r['ER'] for r in rows if r['rule']==rule),
                                      'cases':sum(r['rule']==rule for r in rows),
                                      'ER':float(np.mean([r['ER'] for r in rows if r['rule']==rule]))} for rule in ('E1','E2')},
             'by_domain':{d:{rule:float(np.mean([r['ER'] for r in rows if r['domain']==d and r['rule']==rule]))
                             for rule in ('E1','E2')} for d in ('real','ai')},
             'by_area':{str(a):{rule:float(np.mean([r['ER'] for r in rows if r['area']==a and r['rule']==rule]))
                                for rule in ('E1','E2')} for a in (.8,.5)},
             'old_S_E_verdicts_changed':False,'new_FA_certification':False,'fresh_shard':'DiffusionDB part-000002',
             'scope':'registered mean-quality point, same dataset collection; no unseen-generator/all-attack claim',
             'artifacts':{n:sha(out/n) for n in ('lock.json','prediction_lock.json','recount.json')},'at':stamp()}
    save(out/'source_differences.json',values);save(out/'analysis.json',summary)
    save(out/'status.json',{'phase':'ANALYSIS_COMPLETE','H_Q':summary['verdict'],'sources':848,'gpu_used':False})
    event('H_Q_ANALYSIS',verdict=summary['verdict'],difference=ci['difference'],low=ci['low'],high=ci['high'],quality_matched=q['matched'])
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify-old','admit','lock','run','recount','analyze'])
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--legacy-data',type=Path)
    parser.add_argument('--reference-archive',type=Path)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=8:parser.error('workers must be 1..8')
    if args.command in ('verify-old','admit') and (args.legacy_data is None or args.reference_archive is None):
        parser.error('legacy-data and reference-archive required')
    if args.command=='admit' and (args.dataset is None or args.selection is None):parser.error('dataset and selection required')
    event('START_CPU_ONLY',command=args.command,project=str(args.project),out=str(args.out),gpu_used=False)
    try:
        {'verify-old':verify_old,'admit':admit,'lock':freeze,'run':run,'recount':recount,'analyze':analyze}[args.command](args)
    except Exception:
        event('FAILED',command=args.command,error=traceback.format_exc())
        raise

if __name__=='__main__':main()
