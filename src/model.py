"""
神经网络模型定义模块
包含CNN和全连接网络两种模型架构
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    创建卷积神经网络(CNN)模型
    
    Args:
        input_shape: 输入图像的形状
        num_classes: 分类数量（0-9共10个数字）
    
    Returns:
        model: 编译好的Keras模型
    """
    model = models.Sequential([
        # 第一个卷积块
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # 第二个卷积块
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # 第三个卷积块
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # 展平层
        layers.Flatten(),
        
        # 全连接层
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        
        # 输出层
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_dense_model(input_shape=(28, 28, 1), num_classes=10):
    """
    创建全连接神经网络模型（简单版本）
    
    Args:
        input_shape: 输入图像的形状
        num_classes: 分类数量
    
    Returns:
        model: 编译好的Keras模型
    """
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def print_model_summary(model):
    """
    打印模型结构摘要
    
    Args:
        model: Keras模型
    """
    model.summary()


if __name__ == "__main__":
    # 测试模型创建
    print("创建CNN模型...")
    cnn_model = create_cnn_model()
    print_model_summary(cnn_model)
    
    print("\n创建全连接模型...")
    dense_model = create_dense_model()
    print_model_summary(dense_model)

