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


def create_data_augmentation(rotation_range=10, width_shift_range=0.1, 
                             height_shift_range=0.1, zoom_range=0.1, 
                             shear_range=0.1, fill_mode='nearest'):
    """
    创建数据增强生成器
    用于训练时实时增强数据，提高模型泛化能力
    
    Args:
        rotation_range: 随机旋转角度范围（度）
        width_shift_range: 水平平移范围（相对于总宽度）
        height_shift_range: 垂直平移范围（相对于总高度）
        zoom_range: 随机缩放范围
        shear_range: 剪切变换范围
        fill_mode: 填充模式（'nearest', 'constant', 'reflect', 'wrap'）
    
    Returns:
        ImageDataGenerator: 数据增强生成器
    """
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
        shear_range=shear_range,
        fill_mode=fill_mode
    )
    return datagen


def visualize_augmented_samples(x_data, y_data, num_samples=10):
    """
    可视化数据增强后的样本
    
    Args:
        x_data: 图像数据
        y_data: 标签数据
        num_samples: 要显示的样本数量
    """
    datagen = create_data_augmentation()
    datagen.fit(x_data)
    
    # 获取增强后的样本
    aug_iter = datagen.flow(x_data[:num_samples], y_data[:num_samples], batch_size=num_samples)
    aug_images, aug_labels = next(aug_iter)
    
    plt.figure(figsize=(15, 6))
    
    # 显示原始样本
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        if x_data.ndim == 4:
            plt.imshow(x_data[i].reshape(28, 28), cmap='gray')
        else:
            plt.imshow(x_data[i], cmap='gray')
        
        if y_data.ndim == 2:
            label = np.argmax(y_data[i])
        else:
            label = y_data[i]
        
        plt.title(f'原始: {label}', fontsize=8)
        plt.axis('off')
    
    # 显示增强后的样本
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + num_samples + 1)
        if aug_images.ndim == 4:
            plt.imshow(aug_images[i].reshape(28, 28), cmap='gray')
        else:
            plt.imshow(aug_images[i], cmap='gray')
        
        if aug_labels.ndim == 2:
            label = np.argmax(aug_labels[i])
        else:
            label = aug_labels[i]
        
        plt.title(f'增强: {label}', fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('data/augmented_samples.png', dpi=150, bbox_inches='tight')
    print("增强样本图像已保存到 data/augmented_samples.png")
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

