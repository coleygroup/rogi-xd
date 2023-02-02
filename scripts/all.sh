output=$1
input=${2:-scripts/datasets.all.txt}
featurizers=${3:-'chemberta'}
N=${4:-10000}
repeats=${5:-5}

logfile=logs/`date +%FT%H:%M:%S`.log
mkdir -p `dirname $logfile`

pcmr -vvvv --log $logfile rogi -o $output -f ${featurizers} -N$N --input ${input} -r $repeats
