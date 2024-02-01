#!/usr/bin/env bash

source env.sh
source "${TESTDIR}"/cfg.sh # defines beta, kappa, action, etc.

# add --gauge_obc_x to enable spatial obc
# add seed=${SEED} if fixing seed
${PYTHON} "${TESTDIR}"/make_pion_plots.py \
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
