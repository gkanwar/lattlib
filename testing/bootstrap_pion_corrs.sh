#!/usr/bin/env bash

source env.sh
source ${TESTDIR}/rhmc_cfg.sh # defines beta, kappa, action, etc.

# add --gauge_obc_x to enable spatial obc
# add seed=${SEED} if fixing seed
${PYTHON} ${TESTDIR}/bootstrap_pion_corrs.py \
    --Lx=${LX} \
    --Lt=${LT} \
    --tag=${TAG} \
    --Ncfg=${NCFG} \
    --n_skip=${NSKIP} \
    --n_therm=${NTHERM} \
    --tau=${TAU} \
    --n_leap=${NLEAP} \
    --type=${TYPE} \
    --beta=${BETA} \
    --kappa=${KAPPA} \
