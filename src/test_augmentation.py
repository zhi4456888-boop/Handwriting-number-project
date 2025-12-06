"""
测试数据增强效果
用于可视化数据增强前后的对比
"""

import sys
import numpy as np
from data_loader import load_mnist_data, preprocess_data, visualize_augmentation

if __name__ == "__main__":
    print("加载MNIST数据集...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    print("预处理数据...")
    x_train, y_train, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test
    )
    
    print("\n可视化数据增强效果...")
    print("将显示原始图像和增强后的图像对比")
    visualize_augmentation(x_train, y_train, num_samples=5)
    
    print("\n数据增强测试完成！")

