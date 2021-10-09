# Tensorboard 格式记录
## 文件说明
目录遵从：  
```
Dataset  
  |-Model Name
    |-Hyper Params (epoch,batch,transforms
        |-saved_weights 权重，以训练集最优val_loss保存
            |-权重文件 (epoch,loss)
        |-train 训练集summary、参数分布
        |-val 验证集summary
```
>发布时在`Tensorboard 2.6.0`可以正常阅览。

## 声明
本仓库所有实验结果可以由任何人使用，但本目录下所有记录为本人所有。  
任何人在使用本目录下日志时需引用本仓库地址。