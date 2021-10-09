# ToyNet
随手写的模型，仅有约`0.064M`参数，但是在弱数据集上有“可以接受”的精度。  
主要是分享超参调参记录以及部分训练中可视化数据。  


## 实验结果
> 所有实验都在`Windows 10 Workstation 21H1` `Python 3.9.7` `PyTorch 1.9.1`下完成。
>
> 不使用BN、较小的`Batch Size`、数据集不做标准化处理，都会有相对更高概率在前几个`Epoch`内<span style='color:orange'>无法训练</span>甚至<span style='color:orange'>完全无法训练</span>。
>
> <span style='color:blue'>**`TODO`**</span> `Dropout`和`Weight_Decay`没有深入尝试。

### MNIST
| BN | Epoch | Bach Size | Pytorch Transforms| loss | acc | val_loss | val_acc|
| -- | -- | -- | -- | -- | -- | -- | -- |
|  | 10 | 32 | None | 0.0011 | 0.9998 | 0.0320 | 0.9919 |
| √ | 10 | 32 | None | 0.0036 | 0.9993 | 0.0170 | 0.9937 |
|  | 30 | 32 | None | 0.0000 | 1.0 | 0.0427 | 0.9929 |
| √ | 30 | 32 | None | 0.0004 | 1.0 | 0.0187 | 0.9943 |
|  | 60 | 32 | None | 0.0000 | 1.0 | 0.0603 | 0.9912 |
| √ | 60 | 32 | None | 0.0001 | 1.0 | 0.0219 | **0.9944** |
|  | 60 | 64 | None | 0.0000 | 1.0 | 0.0548 | 0.9924 |
| √ | 60 | 64 | None | 0.0001 | 1.0 | 0.0263 | 0.9923 |

### CIFAR10
| BN | Epoch | Bach Size | Pytorch Transforms| loss | acc | val_loss | val_acc|
| -- | -- | -- | -- | -- | -- | -- | -- |
| | 60 | 32 | None | 0.1001 | 0.9753 | 2.0938 | 0.6476 |
| √ | 60 | 32 | None | 0.0261 | 0.9927 | 1.3708 | 0.7704 |
| | 60 | 32 | Baisc | 0.5510 | 0.8081 | 0.9546 | 0.6968 |
| √ | 60 | 32 | Basic | 0.1881 | 0.9335 | 0.7040 | 0.8028 |
| | 60 | 32 | Normal* | 1.2185 | 0.5674 | 1.0907 | 0.6307 |
| √ | 60 | 32 | Normal* | 0.8537 | 0.7008 | 0.6433 | 0.7803 |
| | 60 | 64 | None | 0.0047 | 0.9999 | 3.2444 | 0.6739 |
| √ | 60 | 64 | None | 0.0089 | 0.9988 | 1.4369 | 0.7665 |
| | 60 | 64 | Baisc | 0.2584 | 0.9140 | 1.0122 | 0.7432 |
| √ | 60 | 64 | Basic | 0.1275 | 0.9581 | 0.8083 | 0.7945 |
| | 60 | 64 | Normal* | 0.9593 | 0.6640 | 0.8077 | 0.7324 |
| √ | 60 | 64 | Normal* | 0.8110 | 0.7173 | 0.6369 | 0.7857 |
|  | 120 | 64 | None | 0.0008 | 1.0 | 4.0485 | 0.6739 |
| √ | 120 | 64 | None | 0.0030 | 0.9993 | 1.9258 | 0.7567 |
|  | 120 | 64 | Basic | 0.1050 | 0.9696 | 1.4695 | 0.7396 |
| √ | 120 | 64 | Basic | 0.0354 | 0.9892 | 1.1726 | 0.7970 |
|  | 120 | 64 | Normal* | 0.9218 | 0.6781 | 0.7641 | 0.7480 |
| √ | 120 | 64 | Normal* | 0.7527 | 0.7395 | 0.5992 | 0.7986 |
|  | 180 | 32 | Normal* | 1.1630 | 0.5867 | 1.0160 | 0.6539 |
| √ | 180 | 32 | Normal* | 0.7565 | 0.7379 | 0.5736 | 0.8047 |
|  | 180 | 64 | Normal* | 0.8900 | 0.6868 | 0.7526 | 0.7519 |
| √ | 180 | 64 | Normal* | 0.7327 | 0.7452 | 0.5907 | 0.8008 |
|  | 180 | 128 | Normal* | 0.7931 | 0.7220 | 0.6485 | 0.7917 |
| √ | 180 | 128 | Normal* | 0.6958 | 0.7587 | 0.5698 | 0.8118 |
|  | 240 | 64 | Normal* | 0.8465 | 0.7033 | 0.7078 | 0.7682 |
| √ | 240 | 64 | Normal* | 0.7030 | 0.7555 | 0.5590 | 0.8106 |
|  | 240 | 128 | Normal* | 0.7511 | 0.7376 | 0.6228 | 0.7969 |
| √ | 240 | 128 | Normal* | 0.6715 | 0.7652 | 0.5391 | 0.8206 |
|  | 240 | 256 | Normal* | 0.7347 | 0.7438 | 0.6481 | 0.7804 |
| √ | 240 | 256 | Normal* | 0.6625 | 0.7694 | 0.5682 | 0.8119 |

\* 最终结果会受到数据增强影响，震荡明显，仅供参考。  

## Pytorch Transforms
```
from torchvision import transforms

IMAGENET_NORAMALIZE: transforms.Compose = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

```

**None:**  
```
transforms.Compose([transforms.ToTensor()])
```
**Basic:**  
```
# train
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    IMAGENET_NORAMALIZE
])

# val
return transforms.Compose([
    transforms.ToTensor(),
    IMAGENET_NORAMALIZE
])
```
**Normal:** `size` 是图片原始尺寸  
> <span style='color:orange'>!!! 收敛震荡警告 !!!</span>
```
# train
transforms.Compose([
    transforms.RandomResizedCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    IMAGENET_NORAMALIZE
])

# val
transforms.Compose([
    transforms.ToTensor(),
    IMAGENET_NORAMALIZE
])
```

## 训练设置
**Optimizer: `SGD`** (`init_lr = .1`, `momentum =.9`)  
**Loss Function: `CrossEntropyLoss`**  
**LR_Sheduler: `CosineAnnealingLR`**  
由于使用了退火学习率，所有实验最终结果直接取最后一个Epoch结果。  
如需查看详细训练记录，请前往`logs`目录。


## 权重
加载权重代码参考：
```
model.load_state_dict(torch.load('PY_WEIGHT_FILE_PATH'))
```