#!/usr/bin/env bash

source env.sh
source "${PRODDIR}"/x4L32mg0p2.sh # defines beta, kappa, action, etc.

# add --gauge_obc_x to enable spatial obc
# add seed=${SEED} if fixing seed
${PYTHON} "${HMCDIR}"/schwinger_propagator.py \
    --Lx=${LX} \
    --Lt=${LT} \
    --tag=${TAG} \
    --Ncfg=${NCFG} \
    --n_skip=${NSKIP} \
    --n_therm=${NTHERM} \
    --tau=${TAU} \
    --n_leap=${NLEAP} \
    --gauge_obc_x \
    --type=${TYPE} \
    --conn_weight=${CONN_WEIGHT} \
    --disc_weight=${DISC_WEIGHT} \
    --beta=${BETA} \
    --kappa=${KAPPA} \
