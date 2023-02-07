input=${1:-scripts/datasets.all.txt}
N=${2:-10000}
repeats=${3:-5}

featurizer=VAE
output=results/data/${featurizer}.txt
model_dir=models/${featurizer}/zinc
pcmr rogi -i ${input} -o $output -f $featurizer -N$N -r $repeats -m ${model_dir} -vvvv --log

featurizers=GIN
output=results/data/${featurizer}.txt
model_dir=models/${featurizer}/zinc
pcmr rogi -i ${input} -o $output -f $featurizer -N$N -r $repeats -m ${model_dir} -vvvv --log
