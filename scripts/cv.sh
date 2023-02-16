#/bin/bash
featurizers=$1
input=${2:-scripts/sample.txt}
N=${3:-10000}
repeats=${4:-1}

if [ -z "$featurizers" ]; then
    featurizers=( descriptor VAE GIN  chemberta chemgpt )
else
    featurizers=( $featurizers )
fi

echo "Running with featurizers: ${featurizers[*]}"

for f in "${featurizers[@]}"; do
    output=results/raw/cv/`basename $input .txt`/${f}.csv
    model_dir=models/${f}/zinc
    pcmr cv -i$input -o$output -f$f -N$N -m$model_dir -vvvv --log
done
