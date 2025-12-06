"""
混淆矩阵分析工具
用于分析模型在5和6之间的混淆情况
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import load_mnist_data, preprocess_data, setup_chinese_font

setup_chinese_font()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path='../models/confusion_matrix.png'):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names or range(10),
                yticklabels=class_names or range(10))
    plt.title('混淆矩阵', fontsize=16, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"混淆矩阵已保存到 {save_path}")
    plt.show()
    
    return cm


def analyze_5_and_6_confusion(model, x_test, y_test):
    """
    专门分析5和6之间的混淆情况
    
    Args:
        model: 训练好的模型
        x_test: 测试图像
        y_test: 测试标签（one-hot编码）
    """
    # 获取预测结果
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # 找出5和6的索引
    indices_5 = np.where(y_true == 5)[0]
    indices_6 = np.where(y_true == 6)[0]
    
    # 统计5和6的混淆情况
    pred_5_as_6 = np.sum((y_true == 5) & (y_pred == 6))
    pred_6_as_5 = np.sum((y_true == 6) & (y_pred == 5))
    
    print("=" * 60)
    print("5和6混淆分析")
    print("=" * 60)
    print(f"真实为5，预测为6的数量: {pred_5_as_6}")
    print(f"真实为6，预测为5的数量: {pred_6_as_5}")
    print(f"5的总数: {len(indices_5)}")
    print(f"6的总数: {len(indices_6)}")
    print(f"5被误判为6的比例: {pred_5_as_6/len(indices_5)*100:.2f}%")
    print(f"6被误判为5的比例: {pred_6_as_5/len(indices_6)*100:.2f}%")
    
    # 可视化混淆的样本
    visualize_confused_samples(model, x_test, y_test, digit1=5, digit2=6, num_samples=10)
    
    return pred_5_as_6, pred_6_as_5


def visualize_confused_samples(model, x_test, y_test, digit1=5, digit2=6, num_samples=10):
    """
    可视化混淆的样本
    
    Args:
        model: 训练好的模型
        x_test: 测试图像
        y_test: 测试标签
        digit1: 第一个数字
        digit2: 第二个数字
        num_samples: 要显示的样本数量
    """
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # 找出混淆的样本
    confused_1_as_2 = np.where((y_true == digit1) & (y_pred == digit2))[0]
    confused_2_as_1 = np.where((y_true == digit2) & (y_pred == digit1))[0]
    
    if len(confused_1_as_2) == 0 and len(confused_2_as_1) == 0:
        print(f"\n没有找到{digit1}和{digit2}之间的混淆样本")
        return
    
    # 选择要显示的样本
    num_1_as_2 = min(num_samples // 2, len(confused_1_as_2))
    num_2_as_1 = min(num_samples // 2, len(confused_2_as_1))
    
    fig, axes = plt.subplots(2, max(num_1_as_2, num_2_as_1), figsize=(15, 6))
    if max(num_1_as_2, num_2_as_1) == 1:
        axes = axes.reshape(2, 1)
    
    # 显示digit1被误判为digit2的样本
    for i in range(num_1_as_2):
        idx = confused_1_as_2[i]
        axes[0, i].imshow(x_test[idx].reshape(28, 28), cmap='gray')
        conf = predictions[idx][digit2]
        axes[0, i].set_title(f'真实:{digit1}→预测:{digit2}\n置信度:{conf:.2%}', 
                            color='red', fontsize=10)
        axes[0, i].axis('off')
    
    # 显示digit2被误判为digit1的样本
    for i in range(num_2_as_1):
        idx = confused_2_as_1[i]
        axes[1, i].imshow(x_test[idx].reshape(28, 28), cmap='gray')
        conf = predictions[idx][digit1]
        axes[1, i].set_title(f'真实:{digit2}→预测:{digit1}\n置信度:{conf:.2%}', 
                            color='red', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'../models/confused_{digit1}_{digit2}_samples.png', dpi=150, bbox_inches='tight')
    print(f"\n混淆样本已保存到 ../models/confused_{digit1}_{digit2}_samples.png")
    plt.show()


def main():
    """主函数"""
    import sys
    
    # 加载模型
    model_path = '../models/mnist_improved_best.h5'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    try:
        model = keras.models.load_model(model_path)
        print(f"已加载模型: {model_path}")
    except Exception as e:
        print(f"无法加载模型: {e}")
        print("请先运行 train.py 或 train_improved.py 训练模型")
        sys.exit(1)
    
    # 加载测试数据
    (_, _), (x_test, y_test) = load_mnist_data()
    _, _, x_test, y_test = preprocess_data(
        np.zeros((1, 28, 28)), np.zeros((1,)), 
        x_test, y_test
    )
    
    # 获取预测结果
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # 绘制完整混淆矩阵
    print("\n生成混淆矩阵...")
    cm = plot_confusion_matrix(y_true, y_pred)
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))
    
    # 专门分析5和6
    print("\n" + "=" * 60)
    analyze_5_and_6_confusion(model, x_test, y_test)
    
    # 显示5和6的混淆矩阵细节
    print("\n5和6的详细混淆情况:")
    print(f"5→5: {cm[5, 5]}, 5→6: {cm[5, 6]}, 5→其他: {np.sum(cm[5, :]) - cm[5, 5] - cm[5, 6]}")
    print(f"6→6: {cm[6, 6]}, 6→5: {cm[6, 5]}, 6→其他: {np.sum(cm[6, :]) - cm[6, 6] - cm[6, 5]}")


if __name__ == "__main__":
    main()

