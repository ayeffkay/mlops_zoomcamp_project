#!/bin/bash

prefect cloud login --key ${PREFECT_KEY} --workspace ${PREFECT_WORKSPACE}
if [ "$1" = true ] ; then
    prefect agent start --pool ${PREFECT_POOL}
fi