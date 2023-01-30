output=$1
input=${2:-scripts/datasets.all.txt}
featurizers=${3:-'chemberta descriptor'}
N=${4:-10000}
repeats=${5:-5}


pcmr rogi -o $output -f ${featurizers} -N$N --input ${input} -r $repeats