#!/bin/bash

usage() {
    echo "Usage: $(basename "$0") [-h] [-f FEATURIZERS] [-i INPUT] [-n N] [-r] [-v]"
    echo
    echo "   -f FEATURIZERS         the featurizers to use (default='descriptor morgan VAE GIN chemberta CHEMGPT')"
    echo "   -i INPUT               a plaintext file containing a dataset on each line (default=scripts/tdc+guac.txt)"
    echo "   -n N                   the number of data to downsample to, if necessary (default=10000)"
    echo "   -l LENGTH              the length of the embedding for random or morgan"
    echo "   -r                     whether to reinitialize the chemberta and chemgpt models"
    echo "   -v                     whether to use the v1 ROGI formulation"
    echo "   -h                     show the help and exit"
    echo
    return
}

while getopts "hrvi:n:f:l:" arg; do
    case $arg in
        f) featurizers=( $OPTARG );;
        i) input=$OPTARG;;
        n) N=$OPTARG;;
        l) length=$OPTARG;;
        r) reinit=true;;
        v) v1=true;;
        h) usage; exit 0;;
        *) echo "Invalid argument"; usage; exit 1;;
    esac
done

[[ -z "$featurizers" ]] && featurizers=( descriptor morgan VAE GIN  chemberta chemgpt random )
[[ -z "$input" ]] && input="scripts/tdc+guac.txt"
[[ -z "$N" ]] && N=10000
[[ -z "$reinit" ]] && reinit=false

[[ "$reinit" = true ]] && reinit_flag="--reinit" || reinit_flag=""
if [[ "$v1" = true ]]; then
    parent_dir=`basename $input .txt`_v1
    v1_flag="--v1"
else
    parent_dir=`basename $input .txt`_v2
    v1_flag=""
fi


echo "Running with featurizers: ${featurizers[*]}"

for f in "${featurizers[@]}"; do
    if [[ "$f" == "random" ]]; then
        name=${f}${length}.json
    elif [[ "$reinit" = true ]]; then
        name=${f}_reinit.json
    else
        name=${f}.json
    fi
    output=results/raw/cv/${parent_dir}/${name}
    model_dir=models/$f/zinc

    pcmr rogi -i$input -o$output -f$f -N$N -m$model_dir -vvvv --log --cv --cg -l$length \
        ${reinit_flag} ${v1_flag}

    if [[ "$reinit" = true ]]; then
        sed -i -e "s/${f}/${f}_reinit/I" $output 
    fi
done
