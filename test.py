import numpy as np
import matplotlib.pyplot as plt

# 示例二维数据（96x48）
data = np.random.rand(96, 48)  # 请替换为你的96x48数据

# 设置 figsize，确保单元格为正方形
fig, ax = plt.subplots(figsize=(24, 12))  # 根据数据比例设置合适的图像尺寸
# 使用 imshow 画热图，aspect='equal' 确保每个单元格是正方形
cax = ax.imshow(data, cmap='viridis', aspect='equal')

# 在每个方格内添加值的文本
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i, j]:.2f}", ha='center', va='center',
                fontsize=6,  # 减小字体大小
                color="white" if data[i, j] > 0.5 else "black")

# 去掉坐标轴
ax.set_xticks([])
ax.set_yticks([])

# 添加颜色条
plt.colorbar(cax, orientation='vertical', pad=0.02)
plt.show()
plt.waitforbuttonpress()
