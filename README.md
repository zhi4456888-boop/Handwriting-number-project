# 手写数字识别AI项目

这是一个使用深度学习技术训练AI模型来识别手写数字（0-9）的项目。项目使用卷积神经网络(CNN)和全连接网络来识别MNIST手写数字数据集。

## 项目结构

```
├── src/                 # 源代码目录
│   ├── __init__.py      # 包初始化文件
│   ├── data_loader.py   # 数据加载和预处理（包含数据增强）
│   ├── model.py         # 神经网络模型定义（包含改进模型）
│   ├── train.py         # 模型训练脚本（标准版）
│   ├── train_improved.py # 改进的训练脚本（高准确率版）
│   ├── train_focused_5_6.py # 针对5和6混淆问题的训练脚本
│   ├── analyze_confusion.py # 混淆矩阵分析工具
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

#### 标准训练（快速）

进入 `src` 目录，运行标准训练脚本：

```bash
cd src
python train.py
```

或者使用批处理文件（Windows）：
```bash
run_train.bat
```

#### 改进训练（高准确率，推荐）

使用改进的模型架构和数据增强技术：

```bash
cd src
python train_improved.py
```

或者使用批处理文件（Windows）：
```bash
run_train_improved.bat
```

**改进版特点：**
- 更深的CNN网络结构
- BatchNormalization层提高训练稳定性
- 数据增强（旋转、平移、缩放等）
- 更长的训练轮数和更好的学习率调度
- **预期准确率：99.2-99.5%+**

#### 针对5和6混淆问题的训练（高级）

如果发现模型在区分5和6时表现不佳，可以使用专门的训练脚本：

```bash
cd src
python train_focused_5_6.py attention
```

或者使用批处理文件（Windows）：
```bash
run_train_focused.bat
```

**特点：**
- 使用注意力机制或焦点损失
- 类别权重（给5和6更高权重）
- 针对性的数据增强
- **预期5和6混淆率：降至0.3-0.5%以下**

训练脚本会：
- 自动下载MNIST数据集
- 预处理数据
- 训练模型（标准版或改进版）
- 保存最佳模型和最终模型到 `models/` 目录
- 生成训练历史图表

### 3. 模型评估

#### 基本评估

运行预测脚本查看模型性能：

```bash
cd src
python predict.py
```

或者指定模型路径：

```bash
python predict.py ../models/mnist_cnn_best.h5
```

#### 混淆矩阵分析（推荐）

分析模型在各类别上的表现，特别关注5和6的混淆情况：

```bash
cd src
python analyze_confusion.py
```

或者使用批处理文件（Windows）：
```bash
run_analyze_confusion.bat
```

这个工具会：
- 生成完整的混淆矩阵热力图
- 专门分析5和6之间的混淆情况
- 可视化混淆的样本图像
- 显示详细的分类报告（精确率、召回率、F1分数）

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

### 改进的CNN模型（推荐，高准确率）
- 3个卷积块，每个包含2个卷积层
- BatchNormalization层加速训练
- 128个卷积核的深层特征提取
- 2个全连接层（256和128个神经元）
- 多层Dropout正则化（0.25-0.5）
- 数据增强支持
- **预期准确率：99.2-99.5%+**

### 注意力机制模型（针对5和6混淆问题）
- 基于改进CNN架构
- 通道注意力机制自动学习重要特征
- 类别权重（5和6增加50%权重）
- 针对性数据增强
- **预期5和6混淆率：0.3-0.5%以下**

### 焦点损失模型（针对难分类样本）
- 基于改进CNN架构
- 焦点损失自动关注难分类样本
- 减少易分类样本的权重
- **预期准确率：99.3-99.6%+**

### 标准CNN模型
- 3个卷积层 + 池化层
- 全连接层 + Dropout
- 输出层（10个类别）
- **预期准确率：98-99%**

### 全连接模型
- 2个隐藏层（512和256个神经元）
- Dropout正则化
- 输出层（10个类别）
- **预期准确率：97-98%**

## 数据集

本项目使用MNIST手写数字数据集：
- **训练集**: 60,000张28x28像素的灰度图像
- **测试集**: 10,000张28x28像素的灰度图像
- **类别**: 0-9共10个数字类别

## 技术栈

- **深度学习框架**: TensorFlow 2.x / Keras
- **编程语言**: Python 3.x
- **可视化**: Matplotlib, Seaborn
- **数据处理**: NumPy, Pandas
- **机器学习**: Scikit-learn
- **GUI**: Tkinter
- **图像处理**: PIL, OpenCV

## 预期性能

### 标准CNN模型
- **准确率**: 98-99%
- **训练时间**: 约5-10分钟（取决于硬件）

### 改进的CNN模型（使用数据增强）
- **准确率**: 99.2-99.5%+
- **训练时间**: 约15-25分钟（取决于硬件）
- **模型大小**: 约3-5MB

## 注意事项

1. 首次运行会自动下载MNIST数据集（约11MB）
2. 训练模型需要一定时间，建议使用GPU加速（如果可用）
3. GUI界面需要在支持图形界面的环境中运行
4. 确保有足够的磁盘空间保存模型文件（每个模型约几MB）

## 常见问题

**Q: 如何选择使用哪个模型？**  
A: 
- 快速测试：使用 `train.py` 和标准CNN模型
- 追求高准确率：使用 `train_improved.py` 和改进的CNN模型

**Q: 如何切换模型类型？**  
A: 在 `train.py` 中修改 `model_type` 变量为 `'cnn'`, `'improved_cnn'` 或 `'dense'`

**Q: 改进模型和标准模型有什么区别？**  
A: 改进模型使用更深的网络、BatchNormalization和数据增强，准确率更高但训练时间更长

**Q: 模型训练很慢怎么办？**  
A: 可以减少 `epochs` 参数，或使用GPU版本的TensorFlow。标准模型训练更快

**Q: GUI无法启动？**  
A: 确保已安装tkinter（通常Python自带），在Linux上可能需要安装 `python3-tk`

**Q: 如何进一步提高准确率？**  
A: 
- 使用改进模型（`train_improved.py`）
- 增加训练轮数（epochs）
- 调整数据增强参数
- 尝试集成学习（训练多个模型并平均预测）

**Q: 模型在区分5和6时表现不佳怎么办？**  
A: 
- 运行 `analyze_confusion.py` 分析混淆情况
- 使用 `train_focused_5_6.py` 训练专门针对5和6的模型
- 推荐使用注意力机制模型：`python train_focused_5_6.py attention`
- 详细说明请参考 `FIX_5_6_CONFUSION.md`

**Q: 如何分析模型的混淆情况？**  
A: 
- 运行 `python analyze_confusion.py [模型路径]`
- 会生成混淆矩阵热力图和详细报告
- 特别分析5和6之间的混淆样本

## 相关文档

- **IMPROVEMENTS.md** - 模型改进详细说明
- **FIX_5_6_CONFUSION.md** - 5和6混淆问题解决方案
- **QUICKSTART.md** - 快速开始指南

## 许可证

本项目仅供学习和研究使用。


