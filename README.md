# MG-attack
This is the code of [Adversarial Label Poisoning Attack on Graph Neural Networks via Label Propagation (ECCV 2022)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650223.pdf). 

## Basic Environment
* Python == 3.7.13

* Pytorch == 1.11.0

## Datasets
All of the datasets we used are in the data folder.

## Run
```
python train_GCN.py

python train_GAT.py

python train_GraphSAGE.py
```

## References
https://github.com/DongHande/PT_propagation_then_training

https://github.com/EpoAtk/EpoAtk

https://github.com/xuanqing94/AdvSSL
