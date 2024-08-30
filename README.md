# Positive-Unlabeled Learning with Label Distribution Alignment (TPAMI 2023)

This is a PyTorch implementation of [PULDA](https://ieeexplore.ieee.org/document/10264106), which is an extension of [Dist-PU](https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective).


## Environments

* python>=3.7
* torch>=1.8.1
* torchvision>=0.9.1
* numpy>=1.19.2
* sklearn>=0.24.1

## Data Preparation
1. Download *CIFAR-10 python version* (163MB) from http://www.cs.utoronto.ca/~kriz/cifar.html to your machine.
2. Decompress the downloaded file *cifar-10-python.tar.gz* from the first step.
3. Usually the second step would result in a new directory like '*/cifar-10-batches-py/' with files in it including:
- data_batch_[1-5]
- test_batch
- batches.meta
- readme.html

## Command
python train.py --device *GPUID* --datapath *DATAPATH*

### Citation

If you use the code of this repository, please cite our paper:
```
@ARTICLE{jiang22maxmatch,
  author={Yangbangyan Jiang and
          Qianqian Xu and
          Yunrui Zhao and
          Zhiyong Yang and
          Peisong Wen and
          Xiaochun Cao and
          Qingming Huang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Positive-Unlabeled Learning with Label Distribution Alignment},
  volume={45},
  number={12},
  pages={15345--15363},
  year={2023}
}
```
