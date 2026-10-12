"""Extract UniLION BEV features for cot-scene-token (P-001; design stage1 sections 7, 8.1, 8.3).

Feature: pts_neck output, 384 x 180 x 180 over +-54 m. The test pipeline's ToEgo moves points to the
ego frame at the CAM_FRONT timestamp, so the BEV lives in that frame (x forward, y left), not in the
LiDAR frame the design assumed. It is average-pooled to --pool x --pool and stored as float16.

Subcommands (run from the repository root with PYTHONPATH=.):
  init      {split}.f16.npy (memmap [N, 384, P, P]), {split}_index.json, {split}_meta.json in --out-dir.
            Row i is the i-th sample of the dataset, i.e. of the infos sorted by timestamp.
  run       fill the rows of one shard. A shard is a contiguous block of whole scenes in dataset order, so
            the streaming temporal memory sees every scene from its first frame, as in evaluation.
  finalize  check that every row was written once, values are finite and tokens match; {split}_check.json
  verify    re-infer random rows from the start of their scenes and compare with the cache; {split}_verify.json
  axis      detection heatmap at GT box centres under each candidate (row, column) reading; axis_check.json
  pick      choose whole scenes with a seed for a partial cache (stage1 8.2 probe); {split}_rows.json
With --rows-file (from pick), init and run build a partial cache: row i holds the i-th picked row of the full
dataset order, and the meta records it as source_row.
"""
import argparse
import json
import os
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from pyquaternion import Quaternion
from torch.utils.data import DataLoader, Subset

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector
import projects.mmdet3d_plugin  # noqa: F401  (registers UniLION modules)

C_BEV, FULL, HALF, CELL = 384, 180, 54.0, 0.6


def build_dataset_for(cfg_path, split):
    cfg = Config.fromfile(cfg_path)
    cfg.data.test.ann_file = os.path.join(cfg.data_root, 'nuscenes_infos_%s.pkl' % split)
    cfg.model.train_cfg = None
    return cfg, build_dataset(cfg.data.test)


def pose(rot, trans):
    m = np.eye(4)
    m[:3, :3] = Quaternion(rot).rotation_matrix
    m[:3, 3] = trans
    return m


def bev2global(info):
    cam = info['cams']['CAM_FRONT']
    return pose(cam['ego2global_rotation'], cam['ego2global_translation'])


def lidar2global(info):
    return pose(info['ego2global_rotation'], info['ego2global_translation']) @ \
        pose(info['lidar2ego_rotation'], info['lidar2ego_translation'])


def shards(infos, n):
    """Split dataset rows into n contiguous blocks of whole scenes with similar sample counts."""
    starts = [i for i in range(len(infos)) if i == 0 or infos[i]['scene_token'] != infos[i - 1]['scene_token']]
    ends = starts[1:] + [len(infos)]
    blocks = [[] for _ in range(n)]
    for s, e in zip(starts, ends):
        k = min(n - 1, int(n * (s + e) / 2 / len(infos)))  # block of the scene's midpoint
        blocks[k].extend(range(s, e))
    return blocks


def find_token(obj):
    if isinstance(obj, dict):
        if 'token' in obj:
            return obj['token']
        objs = obj.values()
    elif isinstance(obj, (list, tuple)):
        objs = obj
    elif hasattr(obj, 'data'):
        objs = [obj.data]
    else:
        return None
    for o in objs:
        t = find_token(o)
        if t is not None:
            return t
    return None


def load_model(cfg, checkpoint):
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location='cpu', strict=False, logger=None)
    sd = torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict']
    missing = sorted(set(model.state_dict()) - set(sd))
    if missing:
        raise RuntimeError('model parameters not in checkpoint: %s' % missing[:10])
    return MMDataParallel(model.cuda(), device_ids=[0]).eval()


def loader(ds, rows, workers):
    return DataLoader(Subset(ds, rows), batch_size=1, shuffle=False, num_workers=workers,
                      collate_fn=partial(collate, samples_per_gpu=1))


def selected(infos, rows_file):
    """Dataset rows a cache holds, in dataset order: all of them, or those listed by pick."""
    if not rows_file:
        return list(range(len(infos)))
    sel = json.load(open(rows_file))
    rows = sel['rows']
    assert rows == sorted(rows) and all(infos[r]['token'] == t for r, t in zip(rows, sel['tokens'])), rows_file
    return rows


def cmd_pick(a):
    """Whole scenes in a seeded random order, up to --fraction of the scenes or --frames keyframes. A row depends on
    the earlier frames of its scene through the streaming memory, so a partial cache is built from scene starts.
    With --frames the picked rows are cut to exactly that many in dataset order."""
    _, ds = build_dataset_for(a.config, a.split)
    infos = ds.data_infos
    scenes = []
    for i, inf in enumerate(infos):
        if i == 0 or inf['scene_token'] != infos[i - 1]['scene_token']:
            scenes.append([])
        scenes[-1].append(i)
    order = np.random.default_rng(a.seed).permutation(len(scenes))
    chosen = []
    for k in order:
        if a.fraction and len(chosen) >= round(a.fraction * len(scenes)):
            break
        if a.frames and sum(len(scenes[c]) for c in chosen) >= a.frames:
            break
        chosen.append(int(k))
    rows = sorted(r for c in chosen for r in scenes[c])
    if a.frames:
        rows = rows[:a.frames]
    os.makedirs(a.out_dir, exist_ok=True)
    out = dict(split=a.split, seed=a.seed, fraction=a.fraction, frames=a.frames,
               scenes=sorted({infos[r]['scene_token'] for r in rows}), rows=rows,
               tokens=[infos[r]['token'] for r in rows])
    path = os.path.join(a.out_dir, '%s_rows.json' % a.split)
    if os.path.exists(path):
        raise SystemExit('%s exists' % path)
    json.dump(out, open(path, 'w'))
    print('pick %s: %d of %d scenes, %d rows -> %s' % (a.split, len(out['scenes']), len(scenes), len(rows), path))


def cmd_init(a):
    _, ds = build_dataset_for(a.config, a.split)
    infos = ds.data_infos
    sel = selected(infos, a.rows_file)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, '%s.f16.npy' % a.split)
    if os.path.exists(path):
        raise SystemExit('%s exists; remove it to start over' % path)
    np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=(len(sel), C_BEV, a.pool, a.pool)).flush()
    json.dump({infos[r]['token']: i for i, r in enumerate(sel)},
              open(os.path.join(a.out_dir, '%s_index.json' % a.split), 'w'))
    rows, seen, keep = [], {}, set(sel)
    for r, inf in enumerate(infos):
        seen[inf['scene_token']] = seen.get(inf['scene_token'], 0) + 1
        if r not in keep:
            continue
        rows.append(dict(row=len(rows), source_row=r, token=inf['token'], scene_token=inf['scene_token'],
                         scene_name=inf['scene_name'], idx_in_scene=seen[inf['scene_token']] - 1,
                         timestamp=inf['timestamp'], prev=inf['prev'], next=inf['next'],
                         bev2global=bev2global(inf).tolist(), lidar2global=lidar2global(inf).tolist()))
    json.dump(dict(frame='ego at CAM_FRONT timestamp (UniLION ToEgo); x forward, y left, z up',
                   range_m=[-HALF, HALF], full_grid=FULL, pool=a.pool, channels=C_BEV,
                   feature='pts_neck output, adaptive_avg_pool2d', rows_file=a.rows_file, rows=rows),
              open(os.path.join(a.out_dir, '%s_meta.json' % a.split), 'w'))
    print('init %s: %d rows, %d scenes -> %s' % (a.split, len(rows), len({x['scene_token'] for x in rows}), path))


def cmd_run(a):
    cfg, ds = build_dataset_for(a.config, a.split)
    infos = ds.data_infos
    sel = selected(infos, a.rows_file)
    pos = {r: i for i, r in enumerate(sel)}  # dataset row -> cache row
    rows = [sel[i] for i in shards([infos[r] for r in sel], a.num_shards)[a.shard]]
    mm = np.load(os.path.join(a.out_dir, '%s.f16.npy' % a.split), mmap_mode='r+')
    assert mm.shape == (len(sel), C_BEV, a.pool, a.pool), mm.shape
    model = load_model(cfg, a.checkpoint)
    feats = {}
    model.module.pts_neck.register_forward_hook(
        lambda _m, _i, out: feats.__setitem__('bev', out[0] if isinstance(out, (list, tuple)) else out))
    done, t0 = [], time.time()
    with torch.no_grad():
        for j, data in enumerate(loader(ds, rows, a.workers)):
            row = rows[j]
            token = find_token(data['img_metas'])
            if token != infos[row]['token']:
                raise RuntimeError('row %d: loader token %s != info token %s' % (row, token, infos[row]['token']))
            model(return_loss=False, rescale=True, **data)
            bev = feats.pop('bev')
            assert tuple(bev.shape) == (1, C_BEV, FULL, FULL), tuple(bev.shape)
            mm[pos[row]] = F.adaptive_avg_pool2d(bev.float(), a.pool)[0].half().cpu().numpy()
            done.append(pos[row])
            if (j + 1) % 500 == 0:
                mm.flush()
                print('shard %d/%d: %d/%d rows, %.1f rows/s' % (a.shard, a.num_shards, j + 1, len(rows),
                                                               (j + 1) / (time.time() - t0)), flush=True)
    mm.flush()
    json.dump(dict(shard=a.shard, num_shards=a.num_shards, rows=done, seconds=time.time() - t0),
              open(os.path.join(a.out_dir, '%s_done.%d-of-%d.json' % (a.split, a.shard, a.num_shards)), 'w'))
    print('shard %d/%d done: %d rows in %.0fs' % (a.shard, a.num_shards, len(done), time.time() - t0))


def cmd_finalize(a):
    meta = json.load(open(os.path.join(a.out_dir, '%s_meta.json' % a.split)))
    index = json.load(open(os.path.join(a.out_dir, '%s_index.json' % a.split)))
    n = len(meta['rows'])
    counts = np.zeros(n, dtype=np.int64)
    for f in sorted(os.listdir(a.out_dir)):
        if f.startswith('%s_done.' % a.split):
            for r in json.load(open(os.path.join(a.out_dir, f)))['rows']:
                counts[r] += 1
    mm = np.load(os.path.join(a.out_dir, '%s.f16.npy' % a.split), mmap_mode='r')
    nonfinite, zero_rows, s1, s2 = 0, 0, 0.0, 0.0
    for i in range(0, n, 256):
        x = mm[i:i + 256].astype(np.float32)
        nonfinite += int((~np.isfinite(x)).sum())
        zero_rows += int((np.abs(x).reshape(len(x), -1).max(1) == 0).sum())
        s1 += float(x.sum())
        s2 += float((x * x).sum())
    size = n * C_BEV * mm.shape[2] * mm.shape[3]
    m = s1 / size
    report = dict(split=a.split, rows=n, index_size=len(index),
                  rows_written_once=int((counts == 1).sum()), rows_missing=int((counts == 0).sum()),
                  rows_written_twice=int((counts > 1).sum()), nonfinite_values=nonfinite, all_zero_rows=zero_rows,
                  index_matches_meta=all(index[r['token']] == r['row'] for r in meta['rows']),
                  mean=m, std=float(np.sqrt(max(s2 / size - m * m, 0.0))))
    report['ok'] = (report['rows_written_once'] == n and nonfinite == 0 and zero_rows == 0
                    and report['index_matches_meta'] and len(index) == n)
    json.dump(report, open(os.path.join(a.out_dir, '%s_check.json' % a.split), 'w'), indent=2)
    print(json.dumps(report, indent=2))


RTOL, ATOL = 2.0 ** -10, 2.0 ** -14  # verify: one float16 ulp relative; smallest normal float16 absolute


def compare(new, ref16):
    """new: float32 [C, P, P]; ref16: float16 row it should reproduce."""
    ref = ref16.astype(np.float32)
    diff = np.abs(new - ref)
    return dict(max_abs_diff=float(diff.max()), rel_l2=float(np.linalg.norm(new - ref) / np.linalg.norm(ref)),
                equal_after_fp16_rounding=float((new.astype(np.float16) == ref16).mean()),
                within_tolerance=bool((diff <= RTOL * np.abs(ref) + ATOL).all()))


def reset_memory(model):
    m = model.module
    m.prev_token, m.prev_features, m.prev_voxel_coords, m.prev_timestamps = [], [], [], []


def cmd_verify(a):
    """Stage1 8.3: re-infer --samples random rows and compare with the cache. The streaming memory makes a row
    depend on earlier frames of its scene, so each picked row is re-run from the first frame of its scene, with the
    memory cleared. With --repeats R > 1 every scene is re-run R times and the re-runs are also compared with each
    other, to tell GPU non-determinism from a wrong cache row; the result goes to {split}_verify_r{R}.json."""
    cfg, ds = build_dataset_for(a.config, a.split)
    infos = ds.data_infos
    mm = np.load(os.path.join(a.out_dir, '%s.f16.npy' % a.split), mmap_mode='r')
    picks = sorted(np.random.default_rng(a.seed).choice(len(infos), a.samples, replace=False).tolist())
    by_scene = {}
    for r in picks:
        by_scene.setdefault(infos[r]['scene_token'], []).append(r)
    model = load_model(cfg, a.checkpoint)
    feats = {}
    model.module.pts_neck.register_forward_hook(
        lambda _m, _i, out: feats.__setitem__('bev', out[0] if isinstance(out, (list, tuple)) else out))
    samples = []
    with torch.no_grad():
        for scene, targets in by_scene.items():
            start = targets[0]
            while start > 0 and infos[start - 1]['scene_token'] == scene:
                start -= 1
            rows = list(range(start, targets[-1] + 1))
            runs = {r: [] for r in targets}
            for _ in range(a.repeats):
                reset_memory(model)
                for j, data in enumerate(loader(ds, rows, a.workers)):
                    model(return_loss=False, rescale=True, **data)
                    bev = feats.pop('bev')
                    if rows[j] in runs:
                        runs[rows[j]].append(F.adaptive_avg_pool2d(bev.float(), a.pool)[0].cpu().numpy())
            for r in targets:
                s = dict(row=r, token=infos[r]['token'], frames_from_scene_start=r - start,
                         cached_std=float(mm[r].astype(np.float32).std()), **compare(runs[r][0], mm[r]))
                if a.repeats > 1:
                    s['vs_cache'] = [compare(x, mm[r]) for x in runs[r]]
                    s['between_reruns'] = [compare(x, runs[r][0].astype(np.float16)) for x in runs[r][1:]]
                    nb = r - 1 if r > start else r + 1  # neighbouring frame: what a misplaced row would look like
                    s['neighbour_frame_rel_l2'] = compare(mm[nb].astype(np.float32), mm[r])['rel_l2']
                samples.append(s)
    out = dict(split=a.split, seed=a.seed, repeats=a.repeats,
               tolerance='|new - cached| <= %g * |cached| + %g' % (RTOL, ATOL),
               samples=samples, ok=all(s['within_tolerance'] for s in samples))
    name = '%s_verify.json' % a.split if a.repeats == 1 else '%s_verify_r%d.json' % (a.split, a.repeats)
    json.dump(out, open(os.path.join(a.out_dir, name), 'w'), indent=2)
    print(json.dumps(out, indent=2))


def cmd_axis(a):
    """Stage1 8.1: which grid reading of the cached BEV puts GT centres on high detection heatmap responses.
    TransFusionHead.forward_single transposes its input (permute(0, 1, 3, 2), transfusion_head.py:252) before
    the heatmap, so the heatmap is transposed back here to the layout of the pts_neck output that is cached."""
    cfg, ds = build_dataset_for(a.config, a.split)
    infos = ds.data_infos
    classes = list(cfg.class_names)
    rows = list(range(a.samples))  # first frames of the first scene(s), in time order
    model = load_model(cfg, a.checkpoint)
    heat = {}
    model.module.det_head.heatmap_head.register_forward_hook(lambda _m, _i, out: heat.__setitem__('h', out))
    readings = {}
    for row_axis in ('x', 'y'):
        for row_sign in (1, -1):
            for col_sign in (1, -1):
                readings['row=%s%s,col=%s%s' % ('+' if row_sign > 0 else '-', row_axis, '+' if col_sign > 0 else '-',
                                                'y' if row_axis == 'x' else 'x')] = (row_axis, row_sign, col_sign)
    sums = {k: [] for k in readings}
    rand_resp = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for j, data in enumerate(loader(ds, rows, a.workers)):
            info = infos[rows[j]]
            model(return_loss=False, rescale=True, **data)
            h = heat.pop('h')[0].sigmoid().transpose(-1, -2).cpu().numpy()  # [classes, 180, 180], BEV layout
            assert h.shape == (len(classes), FULL, FULL), h.shape
            # UniLION's converter already stores gt_boxes in the BEV frame (ego at CAM_FRONT time),
            # unlike the LiDAR-frame infos of mmdet3d; use the centres as they are.
            ok = np.array([n in classes for n in info['gt_names']]) & (info['num_lidar_pts'] > 0)
            ctr = info['gt_boxes'][ok, :3]
            cls = np.array([classes.index(n) for n in info['gt_names'][ok]])
            inside = (np.abs(ctr[:, 0]) < HALF) & (np.abs(ctr[:, 1]) < HALF)
            ctr, cls = ctr[inside], cls[inside]
            for k, (row_axis, rs, cs) in readings.items():
                r_val = ctr[:, 0] if row_axis == 'x' else ctr[:, 1]
                c_val = ctr[:, 1] if row_axis == 'x' else ctr[:, 0]
                ri = np.clip(((rs * r_val + HALF) / CELL).astype(int), 0, FULL - 1)
                ci = np.clip(((cs * c_val + HALF) / CELL).astype(int), 0, FULL - 1)
                sums[k].extend(h[cls, ri, ci].tolist())
            rand_resp.extend(h[cls, rng.integers(0, FULL, len(cls)), rng.integers(0, FULL, len(cls))].tolist())
    result = {k: float(np.mean(v)) for k, v in sums.items()}
    best = max(result, key=result.get)
    out = dict(split=a.split, layout='pts_neck output (heatmap transposed back)', samples=len(rows),
               gt_centres=len(rand_resp), mean_heatmap_at_gt=result,
               mean_heatmap_random_cells=float(np.mean(rand_resp)), best=best,
               frame='ego at CAM_FRONT timestamp; x forward, y left', cell_m=CELL, half_range_m=HALF,
               index_rule='index = floor((sign * coordinate + %.1f) / %.1f)' % (HALF, CELL))
    json.dump(out, open(os.path.join(a.out_dir, 'axis_check_%s.json' % a.split), 'w'), indent=2)
    print(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('cmd', choices=['init', 'run', 'finalize', 'verify', 'axis', 'pick'])
    p.add_argument('--config', default='projects/configs/cot/unilion_lct_det_val.py')
    p.add_argument('--checkpoint')
    p.add_argument('--split', choices=['train', 'val'], required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--pool', type=int, default=45)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--samples', type=int, default=20,
                   help='axis: frames from the start of the split; verify: random rows')
    p.add_argument('--seed', type=int, default=0, help='verify, pick: seed')
    p.add_argument('--repeats', type=int, default=1, help='verify: re-runs per scene')
    p.add_argument('--rows-file', help='init, run: partial cache of the rows picked by pick')
    p.add_argument('--fraction', type=float, help='pick: fraction of the scenes')
    p.add_argument('--frames', type=int, help='pick: number of keyframes')
    a = p.parse_args()
    if a.cmd in ('run', 'verify', 'axis') and not a.checkpoint:
        p.error('--checkpoint is required for %s' % a.cmd)
    dict(init=cmd_init, run=cmd_run, finalize=cmd_finalize, verify=cmd_verify, axis=cmd_axis, pick=cmd_pick)[a.cmd](a)


if __name__ == '__main__':
    main()
