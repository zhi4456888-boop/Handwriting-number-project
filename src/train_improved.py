"""
改进的训练脚本 - 更高准确率版本
使用改进的CNN模型和数据增强技术
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib
import platform
from data_loader import (
    load_mnist_data, 
    preprocess_data, 
    create_data_augmentation,
    setup_chinese_font
)
from model import create_improved_cnn_model, create_cnn_model

# 初始化中文字体
setup_chinese_font()


def train_model_with_augmentation(model, x_train, y_train, x_test, y_test, 
                                  epochs=30, batch_size=128, model_name='mnist_improved'):
    """
    使用数据增强训练模型
    
    Args:
        model: 要训练的模型
        x_train: 训练图像数据
        y_train: 训练标签
        x_test: 测试图像数据
        y_test: 测试标签
        epochs: 训练轮数
        batch_size: 批次大小
        model_name: 模型保存名称
    
    Returns:
        history: 训练历史记录
    """
    # 创建模型保存目录
    os.makedirs('../models', exist_ok=True)
    
    # 创建数据增强生成器
    datagen = create_data_augmentation(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )
    datagen.fit(x_train)
    
    # 定义回调函数
    callbacks = [
        # 保存最佳模型
        keras.callbacks.ModelCheckpoint(
            f'../models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        # 早停机制（更长的耐心值）
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # 学习率衰减（更激进的衰减）
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.000001,
            verbose=1
        ),
        # 学习率调度器（余弦退火）
        keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.001 * np.cos(7 * np.pi * epoch / (2 * epochs))
        )
    ]
    
    print(f"\n开始训练模型（使用数据增强）...")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"训练样本数: {len(x_train)}")
    print(f"测试样本数: {len(x_test)}")
    
    # 使用数据增强训练模型
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # 保存最终模型
    model.save(f'../models/{model_name}_final.h5')
    print(f"\n模型已保存到 ../models/{model_name}_final.h5")
    
    return history


def plot_training_history(history, save_path='../models/training_history_improved.png'):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史记录
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制准确率曲线
    axes[0].plot(history.history['accuracy'], label='训练准确率', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
    axes[0].set_title('模型准确率', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('轮数')
    axes[0].set_ylabel('准确率')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制损失曲线
    axes[1].plot(history.history['loss'], label='训练损失', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='验证损失', linewidth=2)
    axes[1].set_title('模型损失', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('轮数')
    axes[1].set_ylabel('损失')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练历史图表已保存到 {save_path}")
    plt.show()


def main():
    """主函数"""
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("=" * 60)
    print("改进的手写数字识别模型训练")
    print("=" * 60)
    
    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # 预处理数据
    x_train, y_train, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test
    )
    
    # 创建改进的模型
    print("\n创建改进的CNN模型...")
    model = create_improved_cnn_model()
    
    # 显示模型结构
    print("\n模型结构:")
    model.summary()
    
    # 计算模型参数数量
    total_params = model.count_params()
    print(f"\n模型总参数数量: {total_params:,}")
    
    # 训练模型
    history = train_model_with_augmentation(
        model, x_train, y_train, x_test, y_test,
        epochs=30,  # 增加训练轮数
        batch_size=128,
        model_name='mnist_improved'
    )
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 评估模型
    print("\n" + "=" * 60)
    print("模型评估结果")
    print("=" * 60)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # 显示最佳验证准确率
    best_val_acc = max(history.history['val_accuracy'])
    print(f"最佳验证准确率: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    print("\n训练完成！")


if __name__ == "__main__":
    main()

