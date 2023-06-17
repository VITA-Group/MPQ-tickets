echo $0 $1 $2 $3 $4 $5 $6 $7 $8 $9

# $start_iter=$1
# $epoch_iter=$2
# $filename=$3

for i in `seq $1 1 $2`;
  do
    echo "runing $i"
    nohup python iter.py --current_iter $i --datasets tiny_imagenet > iter_$i.out
  done
