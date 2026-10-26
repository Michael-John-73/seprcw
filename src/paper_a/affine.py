"""Train-only two-parameter uniform-pooling control.

The objective gives each image/crop case equal weight, exactly as D3 case_loss.
No observation features, reference message, validation labels, or oracle are used.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from tqdm import tqdm
from .io import log


def affine_objective(theta, z, y, weights, l2):
    """Stable weighted BCE plus l2/2 * ||theta||²; analytic gradient."""
    logits = theta[0] * z + theta[1]
    loss = np.dot(weights, np.logaddexp(0.0, logits) - y * logits)
    residual = weights * (expit(logits) - y)
    gradient = np.array([np.dot(residual, z), residual.sum()]) + l2 * theta
    return float(loss + 0.5 * l2 * np.dot(theta, theta)), gradient


def fit_uniform_affine(paths, load_bank, settings):
    """Stream observations; production objectives and reductions run on CUDA."""
    from .compute import cuda_enabled, tensor
    import torch
    use_cuda = cuda_enabled()
    started = time.perf_counter()
    zz, yy, ww = [], [], []
    for path in tqdm(paths, desc="D4 train-only case aggregates", file=__import__('sys').stdout):
        b = load_bank(path)
        if b['meta']['split'] != 'train':
            raise RuntimeError('D4 fitting permits train banks only')
        n = tensor(b['counts']) if use_cuda else np.asarray(b['counts'])
        valid = n > 0
        if not valid.any():
            raise RuntimeError('D4 train case without observed bits')
        if use_cuda:
            summed = torch.zeros(48, device="cuda", dtype=torch.float64)
            summed.scatter_add_(0, tensor(b['index'], torch.long), tensor(b['soft']))
        else:
            summed = np.bincount(b['index'], weights=b['soft'], minlength=48)
        zz.append(summed[valid] / n[valid])
        yy.append((tensor(b['bits']) if use_cuda else np.asarray(b['bits']))[valid])
        count = int(valid.sum())
        ww.append(torch.full((count,), 1.0 / count, device="cuda", dtype=torch.float64)
                  if use_cuda else np.full(count, 1.0 / count))
    if not zz:
        raise RuntimeError('D4 requires nonempty train banks')
    concatenate = torch.cat if use_cuda else np.concatenate
    z, y, w = concatenate(zz), concatenate(yy), concatenate(ww) / len(zz)
    valid_data = (bool(torch.isfinite(z).all() and ((y == 0) | (y == 1)).all()
                       and torch.unique(y).numel() == 2) if use_cuda else
                  bool(np.isfinite(z).all() and np.isin(y, [0, 1]).all() and len(np.unique(y)) == 2))
    if not valid_data:
        raise RuntimeError('D4 requires finite train evidence and both bit classes')
    def objective(theta, z, y, weights, l2):
        if not use_cuda:
            return affine_objective(theta, z, y, weights, l2)
        t = tensor(theta)
        logits = t[0] * z + t[1]
        loss = (weights * (torch.nn.functional.softplus(logits) - y * logits)).sum()
        residual = weights * (logits.sigmoid() - y)
        gradient = torch.stack(((residual * z).sum(), residual.sum())) + l2 * t
        return float(loss + 0.5 * l2 * t.square().sum()), gradient.cpu().numpy()
    l2 = settings['affine_l2']
    progress = tqdm(total=settings['affine_max_iter'], desc='D4 L-BFGS-B iterations', file=__import__('sys').stdout)
    try:
        result = minimize(objective, np.array([1.0, 0.0]), args=(z,y,w,l2),
            jac=True, method='L-BFGS-B', bounds=[(1e-4,None),(None,None)],
            callback=lambda x: progress.update(1),
            options={'maxiter': settings['affine_max_iter'], 'gtol': settings['affine_gtol'],
                     'ftol': settings['affine_ftol'], 'maxls':20, 'maxfun':15000})
        progress.total = progress.n
        progress.refresh()
    finally:
        progress.close()
    value, gradient = objective(result.x, z,y,w,l2)
    # At the positive-temperature boundary only the feasible descent direction matters.
    projected = gradient.copy()
    if result.x[0] <= 1e-4 + 1e-12:
        projected[0] = min(gradient[0],0.0)
    kkt = float(np.max(np.abs(projected)))
    tolerance = max(10 * settings['affine_gtol'], 1e-6)
    healthy = bool(result.success and np.isfinite(result.x).all() and np.isfinite(value) and kkt <= tolerance)
    report = {'temperature':float(result.x[0]), 'bias':float(result.x[1]),
        'training_healthy':healthy, 'parameter_count':2, 'training_role':'train',
        'cases':len(zz), 'observed_bits':len(z), 'objective':value, 'projected_gradient_max':kkt,
        'kkt_tolerance':tolerance, 'solver_success':bool(result.success),
        'solver_message':str(result.message), 'iterations':int(result.nit),
        'l2':l2, 'elapsed_seconds':time.perf_counter()-started,
        'compute_device':'cuda' if use_cuda else 'cpu',
        'solver_options':{k:settings[k] for k in ['affine_max_iter','affine_gtol','affine_ftol']}}
    log('D4 fit', **report)
    if not healthy:
        raise RuntimeError('D4 optimizer/KKT check failed: ' + str(report))
    return report


def validate_affine(value):
    if not isinstance(value, dict) or value.get('training_healthy') is not True or value.get('training_role') != 'train':
        raise RuntimeError('D4 requires a healthy train-fitted affine state')
    if not np.isfinite([value.get('temperature',np.nan),value.get('bias',np.nan)]).all() or value['temperature'] < 1e-4:
        raise RuntimeError('invalid D4 temperature/bias')
    return value
