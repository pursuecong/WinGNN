# WinGNN: Dynamic Graph Neural Networks with Random Gradient Aggregation Window

## Requirements
  * PyTorch 1.12.1
  * dgl 0.9.1

## How to run
python main.py --dataset uci-msg --lr 0.01 --maml_lr 0.008 --drop_rate 0.16 --window_num 8

## Acknowledgement

Our source code is built heavily based on the code of Roland (https://github.com/snap-stanford/roland)
