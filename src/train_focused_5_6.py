"""
专门针对5和6混淆问题的训练脚本
使用类别权重、焦点损失和针对性的数据增强
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib
import platform
from data_loader import (
    load_mnist_data, 
    preprocess_data, 
    create_data_augmentation,
    setup_chinese_font
)
from model import create_focal_loss_model, create_attention_cnn_model, create_improved_cnn_model

# 初始化中文字体
setup_chinese_font()


def calculate_class_weights(y_train):
    """
    计算类别权重，给5和6更高的权重
    
    Args:
        y_train: 训练标签（one-hot编码）
    
    Returns:
        dict: 类别权重字典
    """
    # 转换为类别索引
    y_classes = np.argmax(y_train, axis=1)
    
    # 计算每个类别的样本数
    class_counts = np.bincount(y_classes)
    total_samples = len(y_classes)
    num_classes = len(class_counts)
    
    # 计算权重（样本数越少，权重越高）
    class_weights = {}
    for i in range(num_classes):
        if class_counts[i] > 0:
            class_weights[i] = total_samples / (num_classes * class_counts[i])
        else:
            class_weights[i] = 1.0
    
    # 特别增加5和6的权重
    class_weights[5] = class_weights[5] * 1.5  # 增加50%权重
    class_weights[6] = class_weights[6] * 1.5  # 增加50%权重
    
    print("类别权重:")
    for i, weight in class_weights.items():
        print(f"  类别 {i}: {weight:.4f}")
    
    return class_weights


def create_focused_data_augmentation():
    """
    创建针对5和6的数据增强
    使用更保守的参数，避免过度变形导致5和6更难区分
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=8,      # 稍微减小旋转范围
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        shear_range=0.08,
        fill_mode='nearest'
    )
    return datagen


def train_focused_model(model, x_train, y_train, x_test, y_test, 
                       epochs=40, batch_size=128, model_name='mnist_focused_5_6',
                       use_class_weights=True, use_focal_loss=False):
    """
    训练专门针对5和6混淆问题的模型
    
    Args:
        model: 要训练的模型
        x_train: 训练图像数据
        y_train: 训练标签
        x_test: 测试图像数据
        y_test: 测试标签
        epochs: 训练轮数
        batch_size: 批次大小
        model_name: 模型保存名称
        use_class_weights: 是否使用类别权重
        use_focal_loss: 是否使用焦点损失（如果模型已使用焦点损失，设为False）
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # 创建模型保存目录
    os.makedirs('../models', exist_ok=True)
    
    # 计算类别权重
    class_weights = None
    if use_class_weights and not use_focal_loss:
        class_weights = calculate_class_weights(y_train)
    
    # 创建数据增强生成器
    datagen = create_focused_data_augmentation()
    datagen.fit(x_train)
    
    # 定义回调函数
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f'../models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=0.000001,
            verbose=1
        )
    ]
    
    print(f"\n开始训练专门针对5和6混淆问题的模型...")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"使用类别权重: {use_class_weights}")
    print(f"使用焦点损失: {use_focal_loss}")
    
    # 训练模型
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # 保存最终模型
    model.save(f'../models/{model_name}_final.h5')
    print(f"\n模型已保存到 ../models/{model_name}_final.h5")
    
    return history


def plot_training_history(history, save_path='../models/training_history_focused.png'):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history['accuracy'], label='训练准确率', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
    axes[0].set_title('模型准确率', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('轮数')
    axes[0].set_ylabel('准确率')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
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


def analyze_5_6_performance(model, x_test, y_test):
    """分析模型在5和6上的表现"""
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # 5和6的准确率
    indices_5 = np.where(y_true == 5)[0]
    indices_6 = np.where(y_true == 6)[0]
    
    acc_5 = np.sum(y_pred[indices_5] == 5) / len(indices_5)
    acc_6 = np.sum(y_pred[indices_6] == 6) / len(indices_6)
    
    # 混淆统计
    pred_5_as_6 = np.sum((y_true == 5) & (y_pred == 6))
    pred_6_as_5 = np.sum((y_true == 6) & (y_pred == 5))
    
    print("\n" + "=" * 60)
    print("5和6的详细性能分析")
    print("=" * 60)
    print(f"数字5的准确率: {acc_5:.4f} ({acc_5*100:.2f}%)")
    print(f"数字6的准确率: {acc_6:.4f} ({acc_6*100:.2f}%)")
    print(f"5被误判为6: {pred_5_as_6} 次")
    print(f"6被误判为5: {pred_6_as_5} 次")
    print(f"5和6之间的总混淆: {pred_5_as_6 + pred_6_as_5} 次")
    
    return acc_5, acc_6, pred_5_as_6, pred_6_as_5


def main():
    """主函数"""
    import sys
    
    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("=" * 60)
    print("专门针对5和6混淆问题的模型训练")
    print("=" * 60)
    
    # 选择模型类型
    model_type = 'attention'  # 'focal', 'attention', 或 'improved'
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    
    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    x_train, y_train, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test
    )
    
    # 创建模型
    if model_type == 'focal':
        print("\n使用焦点损失模型...")
        model = create_focal_loss_model(alpha=0.25, gamma=2.0)
        use_focal_loss = True
    elif model_type == 'attention':
        print("\n使用注意力机制模型...")
        model = create_attention_cnn_model()
        use_focal_loss = False
    else:
        print("\n使用改进的CNN模型（带类别权重）...")
        model = create_improved_cnn_model()
        use_focal_loss = False
    
    model.summary()
    
    # 训练模型
    history = train_focused_model(
        model, x_train, y_train, x_test, y_test,
        epochs=40,
        batch_size=128,
        model_name=f'mnist_focused_5_6_{model_type}',
        use_class_weights=(not use_focal_loss),
        use_focal_loss=use_focal_loss
    )
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 评估模型
    print("\n" + "=" * 60)
    print("模型整体评估")
    print("=" * 60)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # 专门分析5和6
    analyze_5_6_performance(model, x_test, y_test)
    
    print("\n训练完成！")
    print("\n提示：运行以下命令分析混淆矩阵：")
    print("python analyze_confusion.py ../models/mnist_focused_5_6_" + model_type + "_best.h5")


if __name__ == "__main__":
    main()

