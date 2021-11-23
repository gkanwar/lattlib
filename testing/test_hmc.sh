#!/usr/bin/env bash

source env.sh

# define beta, kappa, action, etc.
# call the script

SEED=1
LX=16
LT=16
TAG="testing"
NCFG=1000
NSKIP=10
NTHERM=20
TAU=0.1
NLEAP=20

# "exact_2flav_wilson"
TYPE="two_flavor"
BETA=2.0
KAPPA=1.0
COMPUTE_DIRAC=""

${PYTHON} ${HMCDIR}/schwinger_hmc.py \
    --Lx=${LX} \
    --Lt=${LT} \
    --seed=${SEED} \
    --tag=${TAG} \
    --Ncfg=${NCFG} \
    --n_skip=${NSKIP} \
    --n_therm=${NTHERM} \
    --tau=${TAU} \
    --n_leap=${NLEAP} \
    --type=${TYPE} \
    --beta=${BETA} \
    --kappa=${KAPPA} \
    --compute_dirac=${COMPUTE_DIRAC}
