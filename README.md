# Det_Baseline
### 关于作者
前2届凭证篡改检测竞赛Top1，本届作为主办方成员无法参加，提供一些最基本的方法、思路供community参考和优化
### 关于基线模型
* 训练代码: [code](https://tianchi.aliyun.com/competition/entrance/532267/information) (Baseline.zip)。
* 环境版本: [requirements.txt](https://github.com/qcf-568/Det_Baseline/blob/main/requirements.txt) (直接按上届比赛的实验环境导出的，仅作参考，没验证过现状还能不能用，主要注意用mmdet>=2.0.0)。
* 推理代码：[tta_infer.py](https://github.com/qcf-568/Det_Baseline/blob/main/tta_infer.py) (单模推理脚本，使用了最基础的TTA(水平翻转、多尺度)，加模型集成等可以直接在这个基础上改)。
* 其他情况：提供的基线模型 (SwinL+CascadeR-CNN) 是实际参赛用到的Strong baselines里面指标最低但是速度最快的。
### 基线模型的优化思路
* 骨干网络：对于这个任务，一条结论是较好的骨干网络应该同时具有：1、模拟CNN进行邻域建模，但又2、有灵活的空间特征交互能力（而非静态卷积）这两个特点。SwinTransformer是最基础的例子，可以按这条原则找其他更好的骨干网络。
* 检测头：Detr系列可能会更好（e.g., DINO）。
* 小模块：最常见的是Global context，把整张图进行RoI Align后拼接到每个RoI上。
* 训练数据：利用训练集真实文本自动产生高质量篡改 / 把非篡改区域切出来作为无篡改训练样本。

