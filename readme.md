
# HRNet

## 数据集
**Person Keypoints 数据集**

- 包含 17 个人体关键点，数据来源于 COCO 2017。
- 裁剪出单人图像及其对应的关键点。

### 数据集规模
- **训练集**: xxxx 张图片
- **验证集**: xxx 张图片

[下载链接](https://pan.quark.cn/s/fca45d701e44)


## 预训练模型
- hrnet_w18_pretrained.pth
- hrnet_w30_pretrained.pth
解压后放到`mode_data`文件夹
[下载链接](https://pan.quark.cn/s/b0235b1f041c)

## 关键点检测损失函数

针对关键点检测的损失函数，通常使用均方误差（MSE）来度量预测坐标与真实坐标之间的差异。损失函数可以表示为：

\[
\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} \left( (x_i^{\text{target}} - x_i^{\text{pred}})^2 + (y_i^{\text{target}} - y_i^{\text{pred}})^2 \right)
\]

其中：
- \(N\) 是关键点的总数。
- \(x_i^{\text{target}}\) 和 \(y_i^{\text{target}}\) 是第 \(i\) 个关键点的真实坐标。
- \(x_i^{\text{pred}}\) 和 \(y_i^{\text{pred}}\) 是第 \(i\) 个关键点的预测坐标。

如果引入权重 \(w_i\) 以考虑关键点的重要性，损失函数可以修改为：

\[
\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} w_i \left( (x_i^{\text{target}} - x_i^{\text{pred}})^2 + (y_i^{\text{target}} - y_i^{\text{pred}})^2 \right)
\]

在这里，\(w_i\) 是每个关键点的权重，可能取值在 \(0\) 到 \(1\) 之间，用于指示该关键点的存在性。

## 参考

- [HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification)


- [hrnet-pytorch](https://github.com/bubbliiiing/hrnet-pytorch)

