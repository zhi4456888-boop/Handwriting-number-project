# 快速开始指南

## 第一步：安装依赖

```bash
pip install tensorflow numpy matplotlib scikit-learn pandas pillow opencv-python
```

## 第二步：训练模型

### 方法1：使用批处理文件（Windows）
双击 `run_train.bat`

### 方法2：使用命令行
```bash
cd src
python train.py
```

训练过程：
- 自动下载MNIST数据集（首次运行）
- 训练CNN模型（约10个epoch）
- 保存模型到 `models/` 目录
- 显示训练历史图表

## 第三步：测试识别

### 方法1：使用GUI界面（推荐）
双击 `run_gui.bat` 或运行：
```bash
cd src
python gui.py
```

在画布上绘制数字，点击"识别数字"按钮。

### 方法2：使用命令行
```bash
cd src
python predict.py
```

## 预期结果

- **训练时间**: 5-10分钟（CPU）
- **模型准确率**: 98-99%+
- **模型文件大小**: 约2-5MB

## 提示

1. 首次运行会自动下载MNIST数据集（约11MB）
2. 训练完成后，模型会保存在 `models/` 目录
3. GUI界面需要在图形环境中运行
4. 如果遇到导入错误，确保在 `src` 目录下运行脚本

