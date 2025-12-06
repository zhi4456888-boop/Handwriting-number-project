# 手写数字识别AI项目

这是一个使用深度学习技术训练AI模型来识别手写数字（0-9）的项目。项目使用卷积神经网络(CNN)和全连接网络来识别MNIST手写数字数据集。

## 项目结构

```
├── src/                 # 源代码目录
│   ├── __init__.py      # 包初始化文件
│   ├── data_loader.py   # 数据加载和预处理
│   ├── model.py         # 神经网络模型定义
│   ├── train.py         # 模型训练脚本
│   ├── predict.py       # 预测和推理脚本
│   └── gui.py           # 图形用户界面
├── data/                # 数据集目录（自动下载）
├── models/              # 训练好的模型保存目录
├── notebooks/           # Jupyter notebooks
│   └── exploration.ipynb # 数据探索notebook
├── requirements.txt     # Python依赖包列表
├── .gitignore          # Git忽略文件
└── README.md           # 项目说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

或者单独安装主要依赖：

```bash
pip install tensorflow numpy matplotlib scikit-learn pandas pillow opencv-python
```

## 使用方法

### 1. 数据准备

MNIST数据集会在首次运行时自动下载，无需手动准备。

### 2. 模型训练

进入 `src` 目录，运行训练脚本：

```bash
cd src
python train.py
```

训练脚本会：
- 自动下载MNIST数据集
- 预处理数据
- **使用数据增强技术提升模型准确度**（默认启用）
- 训练CNN模型（默认）或全连接模型
- 保存最佳模型和最终模型到 `models/` 目录
- 生成训练历史图表

#### 数据增强功能

项目已集成数据增强功能，可以通过以下方式提升模型准确度：

**默认配置**（在 `train.py` 中）：
- `use_augmentation = True` - 启用数据增强
- `augment_factor = 2` - 生成2倍训练数据

**数据增强方法包括**：
- 旋转（±15度）
- 平移（±10%）
- 缩放（0.9-1.1倍）
- 弹性变形（模拟手写自然变形）
- 亮度调整
- 添加噪声

**调整数据增强**：
在 `src/train.py` 的 `main()` 函数中修改：
```python
use_augmentation = True   # True启用，False禁用
augment_factor = 2       # 增强倍数（2=2倍数据，3=3倍数据）
```

**预期效果**：
- 训练数据量：从60,000增加到120,000+（augment_factor=2时）
- 准确率提升：通常可提升1-3%
- 泛化能力：对真实手写数字的识别更稳定

### 3. 模型评估

运行预测脚本查看模型性能：

```bash
cd src
python predict.py
```

或者指定模型路径：

```bash
python predict.py ../models/mnist_cnn_best.h5
```

### 4. 使用GUI进行手写识别

启动图形界面，在画布上绘制数字进行识别：

```bash
cd src
python gui.py
```

或者指定模型路径：

```bash
python gui.py ../models/mnist_cnn_best.h5
```

GUI功能：
- 在画布上绘制数字（0-9）
- 点击"识别数字"按钮进行预测
- 显示识别结果和置信度
- 点击"清除画布"重新绘制

### 5. 数据探索

使用Jupyter Notebook探索数据：

```bash
jupyter notebook notebooks/exploration.ipynb
```

## 模型架构

### CNN模型（推荐）
- 3个卷积层 + 池化层
- 全连接层 + Dropout
- 输出层（10个类别）

### 全连接模型
- 2个隐藏层（512和256个神经元）
- Dropout正则化
- 输出层（10个类别）

## 数据集

本项目使用MNIST手写数字数据集：
- **训练集**: 60,000张28x28像素的灰度图像
- **测试集**: 10,000张28x28像素的灰度图像
- **类别**: 0-9共10个数字类别

## 技术栈

- **深度学习框架**: TensorFlow 2.x / Keras
- **编程语言**: Python 3.x
- **可视化**: Matplotlib
- **数据处理**: NumPy, Pandas
- **GUI**: Tkinter
- **图像处理**: PIL, OpenCV

## 预期性能

使用CNN模型，在MNIST测试集上通常可以达到：
- **准确率**: 
  - 无数据增强：98-99%
  - 使用数据增强：99%+（通常可达到99.2-99.5%）
- **训练时间**: 
  - 无数据增强：约5-10分钟
  - 使用数据增强（2倍）：约10-20分钟（取决于硬件）

## 注意事项

1. 首次运行会自动下载MNIST数据集（约11MB）
2. 训练模型需要一定时间，建议使用GPU加速（如果可用）
3. GUI界面需要在支持图形界面的环境中运行
4. 确保有足够的磁盘空间保存模型文件（每个模型约几MB）

## 常见问题

**Q: 如何切换模型类型？**  
A: 在 `train.py` 中修改 `model_type` 变量为 `'cnn'` 或 `'dense'`

**Q: 模型训练很慢怎么办？**  
A: 可以减少 `epochs` 参数，或使用GPU版本的TensorFlow

**Q: GUI无法启动？**  
A: 确保已安装tkinter（通常Python自带），在Linux上可能需要安装 `python3-tk`

## 许可证

本项目仅供学习和研究使用。


