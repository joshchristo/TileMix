## TileMix

**Josiah Soegiharto (27229238), Chee Min Hao (31107176)**

TileMix is a data augmentation technique inspired by CutMix and GridMask. 

This implementation is a modified version of the original [code of CutMix](https://github.com/clovaai/CutMix-PyTorch), that combined with components from the [code of GridMask](https://github.com/dvlab-research/GridMask).

## Getting Started
### Requirements
- Python3
- PyTorch (> 1.0)
- torchvision (> 0.2)
- NumPy

### Train Examples
- CIFAR-100 dataset
- CIFAR-100 dataset
```
python train.py \
--net_type pyramidnet \
--dataset cifar100 \
--depth 200 \
--alpha 240 \
--batch_size 64 \
--lr 0.25 \
--expname PyraNet200 \
--epochs 300 \
--beta 1.0 \
--tilemix_prob 0.5 \
--tilemix_prob 0.5 \
--no-verbose
```
- ImageNet dataset
- ImageNet dataset
```
python train.py \
--net_type resnet \
--dataset imagenet \
--batch_size 256 \
--lr 0.1 \
--depth 50 \
--epochs 300 \
--expname ResNet50 \
-j 40 \
--beta 1.0 \
--tilemix_prob 1.0 \
--tilemix_prob 1.0 \
--no-verbose
```

### Test Examples using Pretrained model
```
python test.py \
--net_type pyramidnet \
--dataset cifar100 \
--batch_size 64 \
--depth 200 \
--alpha 240 \
--pretrained /set/your/model/path/model_best.pth.tar
```
```
python test.py \
--net_type resnet \
--dataset imagenet \
--batch_size 64 \
--depth 50 \
--pretrained /set/your/model/path/model_best.pth.tar
```

## References

- CutMix code: https://github.com/clovaai/CutMix-PyTorch
- GridMask code: https://github.com/dvlab-research/GridMask
## References

- CutMix code: https://github.com/clovaai/CutMix-PyTorch
- GridMask code: https://github.com/dvlab-research/GridMask