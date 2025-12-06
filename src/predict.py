"""
预测和推理脚本
用于使用训练好的模型进行手写数字识别
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib
import platform
from PIL import Image
import cv2

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


def load_trained_model(model_path):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
    
    Returns:
        model: 加载的Keras模型
    """
    print(f"正在加载模型: {model_path}")
    model = keras.models.load_model(model_path)
    print("模型加载成功!")
    return model


def predict_image(model, image, show_image=True):
    """
    对单张图像进行预测
    
    Args:
        model: 训练好的模型
        image: 输入图像（numpy数组，形状为(28, 28)或(28, 28, 1)）
        show_image: 是否显示图像
    
    Returns:
        predicted_digit: 预测的数字
        confidence: 预测的置信度
    """
    # 确保图像形状正确
    if image.ndim == 2:
        image = image.reshape(1, 28, 28, 1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = image.reshape(1, 28, 28, 1)
    elif image.ndim == 3:
        # 如果是RGB图像，转换为灰度
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.reshape(1, 28, 28, 1)
    
    # 归一化
    if image.max() > 1.0:
        image = image.astype('float32') / 255.0
    
    # 进行预测
    predictions = model.predict(image, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = predictions[0][predicted_digit]
    
    if show_image:
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.title(f'预测数字: {predicted_digit}\n置信度: {confidence:.2%}')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.bar(range(10), predictions[0])
        plt.xlabel('数字')
        plt.ylabel('概率')
        plt.title('所有数字的概率分布')
        plt.xticks(range(10))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return predicted_digit, confidence


def predict_from_file(model, image_path):
    """
    从文件加载图像并进行预测
    
    Args:
        model: 训练好的模型
        image_path: 图像文件路径
    
    Returns:
        predicted_digit: 预测的数字
        confidence: 预测的置信度
    """
    # 加载图像
    img = Image.open(image_path).convert('L')  # 转换为灰度图
    img = img.resize((28, 28))  # 调整大小
    img_array = np.array(img)
    
    return predict_image(model, img_array)


def predict_batch(model, images):
    """
    批量预测
    
    Args:
        model: 训练好的模型
        images: 图像数组，形状为(n, 28, 28, 1)
    
    Returns:
        predictions: 预测结果数组
        confidences: 置信度数组
    """
    # 归一化
    if images.max() > 1.0:
        images = images.astype('float32') / 255.0
    
    predictions = model.predict(images, verbose=0)
    predicted_digits = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    return predicted_digits, confidences


def visualize_predictions(model, x_test, y_test, num_samples=10):
    """
    可视化预测结果
    
    Args:
        model: 训练好的模型
        x_test: 测试图像
        y_test: 测试标签（one-hot编码）
        num_samples: 要显示的样本数量
    """
    # 随机选择一些样本
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    predictions, confidences = predict_batch(model, x_test[indices])
    
    # 获取真实标签
    true_labels = np.argmax(y_test[indices], axis=1)
    
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        
        pred = predictions[i]
        true = true_labels[i]
        conf = confidences[i]
        
        # 根据预测是否正确设置标题颜色
        color = 'green' if pred == true else 'red'
        plt.title(f'真实: {true}\n预测: {pred}\n置信度: {conf:.2%}', 
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../models/prediction_samples.png', dpi=150, bbox_inches='tight')
    print("预测结果已保存到 ../models/prediction_samples.png")
    plt.show()


if __name__ == "__main__":
    import sys
    from data_loader import load_mnist_data, preprocess_data
    
    # 加载模型
    model_path = '../models/mnist_cnn_best.h5'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    try:
        model = load_trained_model(model_path)
    except Exception as e:
        print(f"无法加载模型: {e}")
        print("请先运行 train.py 训练模型")
        sys.exit(1)
    
    # 加载测试数据
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    x_train, y_train, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test
    )
    
    # 可视化一些预测结果
    print("\n可视化预测结果...")
    visualize_predictions(model, x_test, y_test, num_samples=10)
    
    # 评估模型
    print("\n评估模型性能...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集准确率: {test_accuracy:.4f}")

