"""Write nuscenes_infos_{train,val}.pkl with UniLION's converter, without the GT database.

The GT database is only used for training-time augmentation; cot-scene-token runs UniLION for
evaluation and BEV feature extraction only (P-001). Sensor paths in the infos are built from
--root-path, so pass the absolute nuScenes root and nothing is written there.

Run from the repository root with PYTHONPATH=. :
    PYTHONPATH=. python tools/cot/make_infos.py --root-path $NUSCENES_ROOT --out-dir data/nuscenes
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--root-path', required=True, help='nuScenes root (v1.0-trainval, samples, sweeps, maps, can_bus)')
parser.add_argument('--out-dir', required=True)
parser.add_argument('--max-sweeps', type=int, default=10)
args = parser.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_converter'))
# nuscenes_converter parses sys.argv when imported; give it an argv it accepts
# (its required='False' makes --out-dir and --db-save-path required; the values are not used here).
saved_argv, sys.argv = sys.argv, ['nuscenes_converter.py', 'nuscenes',
                                  '--out-dir', args.out_dir, '--db-save-path', args.out_dir]
import nuscenes_converter  # noqa: E402
sys.argv = saved_argv

nuscenes_converter.create_nuscenes_infos(
    args.root_path, args.out_dir, args.root_path, 'nuscenes',
    version='v1.0-trainval', max_sweeps=args.max_sweeps)
