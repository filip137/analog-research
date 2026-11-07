#!/usr/bin/env bash
# DRN stretch (main item 2d) on riri: timing (padapter64, ln; deployed view), D1 gap, then timing analog. ~3-6 min total alone on the GPU.
set -u
P=/home/filip/miniconda3/envs/py312/bin/python
R=~/domain-ft-20261108
cd $R/drn/code
mkdir -p $R/logs
$P ../drn_lab.py timing --method padapter64 --view deployed --updates 25 --micro 8 > $R/logs/drn_timing_padapter64.log 2>&1
$P ../drn_lab.py timing --method ln --view deployed --updates 25 --micro 8 > $R/logs/drn_timing_ln.log 2>&1
$P ../drn_lab.py gap --blocks 256 --micro 8 > $R/logs/drn_gap.log 2>&1
$P ../drn_lab.py timing --method analog --view deployed --updates 25 --micro 8 > $R/logs/drn_timing_analog.log 2>&1
echo DRN_DONE >> $R/logs/drn_gap.log
