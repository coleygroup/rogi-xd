output=$1
datasets=${2:-scripts/datasets.all.txt}
featurizers=${3:'chemberta descriptors'}
N=${4:-10000}
repetas=${5:-5}

mapfile -t datasets < $datasets
featurizers=( $featurizers )

pcmr -o $output -f ${featurizers[*]} -N$N --dt ${datasets[*]} -r $repetas