#/bin/bash
featurizers=$1
input=${2:-scripts/tdc.txt}
N=${3:-10000}
repeats=${4:-5}

if [ -z "$featurizers" ]; then
    featurizers=( descriptor chemberta chemgpt GIN VAE )
else
    featurizers=( $featurizers )
fi

echo "Running with featurizers: ${featurizers[*]}"

for f in "${featurizers[@]}"; do
    output=results/raw/`basename $input .txt`/${f}.csv
    model_dir=models/${f}/zinc
    pcmr rogi -i $input -o $output -f $f -N$N -r $repeats -m ${model_dir} -vvvv --log
done