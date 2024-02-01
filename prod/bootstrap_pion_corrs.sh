#!/usr/bin/env bash

source env.sh
source "${PRODDIR}"/x4L32mg0p2.sh # defines beta, kappa, action, etc.

# add --gauge_obc_x to enable spatial obc
# add seed=${SEED} if fixing seed
${PYTHON} "${PRODDIR}"/bootstrap_pion_corrs.py \
    --Lx=${LX} \
    --Lt=${LT} \
    --tag=${TAG} \
    --Ncfg=${NCFG} \
    --n_skip=${NSKIP} \
    --n_therm=${NTHERM} \
    --tau=${TAU} \
    --gauge_obc_x \
    --n_leap=${NLEAP} \
    --type=${TYPE} \
    --beta=${BETA} \
    --kappa=${KAPPA} \
