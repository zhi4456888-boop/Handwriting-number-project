"""
数据加载和预处理模块
用于加载MNIST手写数字数据集并进行预处理
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib
import platform
import cv2
import os

# 配置matplotlib中文字体支持
def setup_chinese_font():
    """配置matplotlib以支持中文显示"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统使用微软雅黑或黑体
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    elif system == 'Darwin':  # macOS
        font_list = ['Arial Unicode MS', 'PingFang SC', 'STHeiti']
    else:  # Linux
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
    
    # 尝试设置中文字体
    for font_name in font_list:
        try:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            return
        except:
            continue
    
    # 如果所有字体都不可用，使用默认设置
    plt.rcParams['axes.unicode_minus'] = False

# 初始化中文字体
setup_chinese_font()


def load_mnist_data():
    """
    加载MNIST数据集
    
    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
    """
    print("正在加载MNIST数据集...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f"训练集大小: {x_train.shape}")
    print(f"测试集大小: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def preprocess_data(x_train, y_train, x_test, y_test):
    """
    预处理数据：归一化、reshape、one-hot编码
    
    Args:
        x_train: 训练图像数据
        y_train: 训练标签
        x_test: 测试图像数据
        y_test: 测试标签
    
    Returns:
        tuple: 预处理后的数据
    """
    # 将图像数据归一化到0-1范围
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 将图像从28x28 reshape为784 (展平)
    # 或者保持28x28用于CNN
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # 将标签转换为one-hot编码
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print("数据预处理完成!")
    print(f"训练集形状: {x_train.shape}")
    print(f"测试集形状: {x_test.shape}")
    print(f"标签形状: {y_train.shape}")
    
    return x_train, y_train, x_test, y_test


def visualize_samples(x_data, y_data, num_samples=10):
    """
    可视化一些样本数据
    
    Args:
        x_data: 图像数据
        y_data: 标签数据
        num_samples: 要显示的样本数量
    """
    plt.figure(figsize=(12, 3))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        # 如果数据是归一化的，需要反归一化显示
        if x_data.ndim == 4:
            plt.imshow(x_data[i].reshape(28, 28), cmap='gray')
        else:
            plt.imshow(x_data[i], cmap='gray')
        
        # 获取真实标签
        if y_data.ndim == 2:  # one-hot编码
            label = np.argmax(y_data[i])
        else:
            label = y_data[i]
        
        plt.title(f'标签: {label}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('data/sample_images.png', dpi=150, bbox_inches='tight')
    print("样本图像已保存到 data/sample_images.png")
    plt.show()


def create_data_augmentation():
    """
    创建数据增强生成器（实时增强）
    
    Returns:
        ImageDataGenerator: 数据增强生成器
    """
    datagen = ImageDataGenerator(
        rotation_range=15,        # 随机旋转角度范围（度）
        width_shift_range=0.1,   # 水平平移范围（图像宽度的比例）
        height_shift_range=0.1,  # 垂直平移范围（图像高度的比例）
        zoom_range=0.1,          # 随机缩放范围
        shear_range=0.1,         # 剪切变换范围
        fill_mode='nearest',     # 填充模式
        brightness_range=[0.8, 1.2],  # 亮度调整范围
    )
    return datagen


def apply_elastic_transform(image, alpha=20, sigma=3, random_state=None):
    """
    应用弹性变形（Elastic Deformation）
    模拟手写时的自然变形
    
    Args:
        image: 输入图像 (28, 28) 或 (28, 28, 1)
        alpha: 变形强度（降低以适应小图像）
        sigma: 平滑度
        random_state: 随机种子
    
    Returns:
        变形后的图像
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    # 确保图像是2D的
    if len(image.shape) == 3:
        img_2d = image.squeeze()
    else:
        img_2d = image
    
    shape = img_2d.shape
    dx = random_state.rand(*shape) * 2 - 1
    dy = random_state.rand(*shape) * 2 - 1
    
    # 应用高斯滤波
    dx = cv2.GaussianBlur(dx, (5, 5), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (5, 5), sigma) * alpha
    
    # 创建坐标网格
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    
    # 应用变形
    transformed = cv2.remap(
        img_2d.astype(np.float32),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    # 确保输出形状与输入一致
    if len(image.shape) == 3:
        return transformed.reshape(shape[0], shape[1], 1)
    return transformed


def augment_data_advanced(x_train, y_train, augment_factor=2):
    """
    高级数据增强：生成更多训练样本
    
    Args:
        x_train: 训练图像
        y_train: 训练标签
        augment_factor: 增强倍数（生成augment_factor倍的数据）
    
    Returns:
        增强后的训练数据
    """
    augmented_images = []
    augmented_labels = []
    
    print(f"开始数据增强，将生成 {augment_factor} 倍的数据...")
    print("这可能需要一些时间，请耐心等待...")
    
    for i in range(len(x_train)):
        # 添加原始数据
        augmented_images.append(x_train[i])
        augmented_labels.append(y_train[i])
        
        # 生成增强数据
        for _ in range(augment_factor - 1):
            img = x_train[i].copy()
            
            # 随机选择增强方法
            method = np.random.choice(['elastic', 'rotate', 'shift', 'zoom', 'noise'])
            
            try:
                if method == 'elastic':
                    img = apply_elastic_transform(img.squeeze()).reshape(28, 28, 1)
                elif method == 'rotate':
                    angle = np.random.uniform(-15, 15)
                    M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
                    img = cv2.warpAffine(img.squeeze(), M, (28, 28)).reshape(28, 28, 1)
                elif method == 'shift':
                    tx = np.random.uniform(-3, 3)
                    ty = np.random.uniform(-3, 3)
                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    img = cv2.warpAffine(img.squeeze(), M, (28, 28)).reshape(28, 28, 1)
                elif method == 'zoom':
                    zoom = np.random.uniform(0.9, 1.1)
                    M = cv2.getRotationMatrix2D((14, 14), 0, zoom)
                    img = cv2.warpAffine(img.squeeze(), M, (28, 28)).reshape(28, 28, 1)
                elif method == 'noise':
                    noise = np.random.normal(0, 0.05, img.shape)
                    img = np.clip(img + noise, 0, 1)
                
                augmented_images.append(img)
                augmented_labels.append(y_train[i])
            except Exception as e:
                # 如果增强失败，使用原始图像
                augmented_images.append(x_train[i])
                augmented_labels.append(y_train[i])
        
        if (i + 1) % 10000 == 0:
            print(f"已处理 {i + 1}/{len(x_train)} 个样本")
    
    print(f"数据增强完成！原始数据: {len(x_train)}, 增强后: {len(augmented_images)}")
    return np.array(augmented_images), np.array(augmented_labels)


def visualize_augmentation(x_data, y_data, num_samples=5):
    """
    可视化数据增强效果
    
    Args:
        x_data: 原始图像数据
        y_data: 标签数据
        num_samples: 要显示的样本数量
    """
    plt.figure(figsize=(15, 6))
    
    for i in range(num_samples):
        # 原始图像
        plt.subplot(2, num_samples, i + 1)
        if x_data.ndim == 4:
            plt.imshow(x_data[i].reshape(28, 28), cmap='gray')
        else:
            plt.imshow(x_data[i], cmap='gray')
        
        if y_data.ndim == 2:
            label = np.argmax(y_data[i])
        else:
            label = y_data[i]
        plt.title(f'原始: {label}')
        plt.axis('off')
        
        # 增强后的图像
        plt.subplot(2, num_samples, i + num_samples + 1)
        img = x_data[i].copy()
        
        # 随机应用一种增强
        method = np.random.choice(['elastic', 'rotate', 'shift', 'zoom', 'noise'])
        try:
            if method == 'elastic':
                img = apply_elastic_transform(img.squeeze()).reshape(28, 28, 1)
            elif method == 'rotate':
                angle = np.random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
                img = cv2.warpAffine(img.squeeze(), M, (28, 28)).reshape(28, 28, 1)
            elif method == 'shift':
                tx = np.random.uniform(-3, 3)
                ty = np.random.uniform(-3, 3)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                img = cv2.warpAffine(img.squeeze(), M, (28, 28)).reshape(28, 28, 1)
            elif method == 'zoom':
                zoom = np.random.uniform(0.9, 1.1)
                M = cv2.getRotationMatrix2D((14, 14), 0, zoom)
                img = cv2.warpAffine(img.squeeze(), M, (28, 28)).reshape(28, 28, 1)
            elif method == 'noise':
                noise = np.random.normal(0, 0.05, img.shape)
                img = np.clip(img + noise, 0, 1)
        except:
            pass
        
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.title(f'增强: {method}')
        plt.axis('off')
    
    plt.tight_layout()
    os.makedirs('data', exist_ok=True)
    plt.savefig('data/augmentation_samples.png', dpi=150, bbox_inches='tight')
    print("数据增强效果图已保存到 data/augmentation_samples.png")
    plt.show()


if __name__ == "__main__":
    # 测试数据加载
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # 预处理数据
    x_train, y_train, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test
    )
    
    # 可视化一些样本
    visualize_samples(x_train, y_train, num_samples=10)

