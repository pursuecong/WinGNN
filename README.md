# WinGNN: Dynamic Graph Neural Networks with Random Gradient Aggregation Window

This repository is our PyTorch implementation of WinGNN.

## Requirements
```shell
pip install -r requirements.txt
```

## How to run
You can run the WinGNN with the following commands:

```shell
# UCI
python main.py --dataset uci-msg --lr 0.01 --maml_lr 0.008 --drop_rate 0.16 --window_num 8
# DBLP
python main.py --dataset dblp --lr 0.007 --maml_lr 0.003 --drop_rate 0.09 --window_num 8
# BitcoinAlpha
python main.py --dataset bitcoinalpha --lr 0.2 --maml_lr 0.003 --drop_rate 0.1 --window_num 8
# BitcoinOTC
python main.py --dataset bitcoinotc --lr 0.003 --maml_lr 0.006 --drop_rate 0.4 --window_num 7
# Reddit-Title
python main.py --dataset reddit_title --lr 0.07 --maml_lr 0.0009 --drop_rate 0.16 --window_num 10
# stackoverflow
python main.py --dataset stackoverflow_M --lr 0.03 --maml_lr 0.001 --drop_rate 0.1 --window_num 8 --num_layers 1 --num_hidden 32 --out_dim 16

```


## Acknowledgement

Our source code and data processing are built heavily based on the code of Roland (https://github.com/snap-stanford/roland).

The data set download address is provided in the paper.

## Reference

If you find this work is helpful to your research, please consider citing our paper:

```
@inproceedings{WinGNN,
  title={WinGNN: Dynamic Graph Neural Networks with Random Gradient Aggregation Window},
  author={Zhu, Yifan and Cong, Fangpeng and Zhang, Dan and Gong, Wenwen and Lin, Qika and Feng, Wenzheng and Dong, Yuxiao and Tang, Jie},
  booktitle={Proceedings of 29th {ACM} {SIGKDD} Conference on Knowledge Discovery and Data Mining},
  year={2023}
}
```