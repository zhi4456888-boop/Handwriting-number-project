"""
模型训练脚本
用于训练手写数字识别模型
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib
import platform
from data_loader import load_mnist_data, preprocess_data, create_data_augmentation
from model import create_cnn_model, create_dense_model, create_improved_cnn_model

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


def train_model(model, x_train, y_train, x_test, y_test, 
                epochs=10, batch_size=128, model_name='mnist_model'):
    """
    训练模型
    
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
    
    # 定义回调函数
    callbacks = [
        # 保存最佳模型
        keras.callbacks.ModelCheckpoint(
            f'../models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # 早停机制
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # 学习率衰减
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    print(f"\n开始训练模型...")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"训练样本数: {len(x_train)}")
    print(f"测试样本数: {len(x_test)}")
    
    # 训练模型
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # 保存最终模型
    model.save(f'../models/{model_name}_final.h5')
    print(f"\n模型已保存到 ../models/{model_name}_final.h5")
    
    return history


def plot_training_history(history, save_path='../models/training_history.png'):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史记录
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制准确率曲线
    axes[0].plot(history.history['accuracy'], label='训练准确率')
    axes[0].plot(history.history['val_accuracy'], label='验证准确率')
    axes[0].set_title('模型准确率')
    axes[0].set_xlabel('轮数')
    axes[0].set_ylabel('准确率')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制损失曲线
    axes[1].plot(history.history['loss'], label='训练损失')
    axes[1].plot(history.history['val_loss'], label='验证损失')
    axes[1].set_title('模型损失')
    axes[1].set_xlabel('轮数')
    axes[1].set_ylabel('损失')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练历史图表已保存到 {save_path}")
    plt.show()


def main():
    """主函数"""
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # 预处理数据
    x_train, y_train, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test
    )
    
    # 选择模型类型：'cnn', 'improved_cnn' 或 'dense'
    model_type = 'cnn'  # 可以改为 'improved_cnn' 使用改进的CNN模型，或 'dense' 使用全连接网络
    use_augmentation = False  # 是否使用数据增强（改进模型建议开启）
    
    if model_type == 'improved_cnn':
        print("\n使用改进的CNN模型...")
        model = create_improved_cnn_model()
        use_augmentation = True  # 改进模型默认使用数据增强
    elif model_type == 'cnn':
        print("\n使用CNN模型...")
        model = create_cnn_model()
    else:
        print("\n使用全连接模型...")
        model = create_dense_model()
    
    # 显示模型结构
    model.summary()
    
    # 训练模型
    if use_augmentation:
        # 使用数据增强训练
        datagen = create_data_augmentation(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            fill_mode='nearest'
        )
        datagen.fit(x_train)
        
        # 更新回调函数以支持更长的训练
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                f'../models/mnist_{model_type}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.000001,
                verbose=1
            )
        ]
        
        print(f"\n使用数据增强训练模型...")
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=128),
            steps_per_epoch=len(x_train) // 128,
            epochs=20,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        model.save(f'../models/mnist_{model_type}_final.h5')
    else:
        # 标准训练
        history = train_model(
            model, x_train, y_train, x_test, y_test,
            epochs=10,
            batch_size=128,
            model_name=f'mnist_{model_type}'
        )
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 评估模型
    print("\n评估模型...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()

