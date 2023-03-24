#/bin/bash

featurizers=$1
input=${2:-scripts/sample.txt}
N=${3:-10000}

if [ -z "$featurizers" ]; then
    featurizers=( descriptor vae gin chemberta chemgpt )
else
    featurizers=( $featurizers )
fi

echo "Running with featurizers: ${featurizers[*]}"

for f in "${featurizers[@]}"; do
    output=results/raw/cg/`basename $input .txt`/${f}.json
    model_dir=models/$f/zinc
    pcmr rogi -i$input -o$output -f$f -N$N -m$model_dir -vvvv --log --cg -r5
done
