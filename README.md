## 📦 Run this project
Ensure that `cuda` is supported by your machine. Try `assert torch.cuda.is_available() == True`

Start your journey via `make start`. You can read detailed code to understand how it works.

### Advanced usage:

To avoid storage problem and optimizer, you should run a `bash` file to run.

Change `seq 0 1 30` to `seq $start 1 $end` to resume your training!

```
for i in `seq 0 1 30`;
  do
    echo "runing $i"
    nohup python3 iter.py --current_iter $i --datasets cifar10 --mask_type minimal --model resnet20 > iter_$i.out
  done
```

Note: if you want to run multiple experiments, you should copy this folder several times and run them seperetely to avoid overwriting model and mask.


## 📑 Project Tree
```
|-- MPQ-ticket
    |-- start.sh              [use this] helper(caller) to start mission using iter.sh
    |-- iter.sh               [shouldn't be used directly] iteratively mission caller,
    |                                                     this script should only be called by `start.sh`
    |
    |-- iter.py               iteratively style quantize-lth to resnet20
    |
    |-- models
    |   |-- resnet
    |   |   |-- resnet.py
    |   |   |-- resnet_q.py
    |   |-- vgg
    |       |-- vgg.py
    |       |-- vgg_q.py
    |-- lth.py                lottery hypothesis ticket
    |-- quantization.py       quantization module
    |
    |-- log.py                log module
    |-- utils.py              utils module (including safe_saver and mask stats helper)
    |
    |-- data                  default folder for data store
    |-- net
    |   |-- arch.py           weight init tool + dataset + loader
    |-- save
    |   |-- model             default store folder for model
    |   |-- mask              default store folder for mask
    |-- README.md             this file
    |-- __init__.py
```
