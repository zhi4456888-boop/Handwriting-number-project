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


def create_improved_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    创建改进的CNN模型（更高准确率）
    
    改进点：
    - 添加BatchNormalization加速训练并提高稳定性
    - 更深的网络结构
    - 更好的正则化策略
    - 使用padding='same'保持特征图尺寸
    
    Args:
        input_shape: 输入图像的形状
        num_classes: 分类数量（0-9共10个数字）
    
    Returns:
        model: 编译好的Keras模型
    """
    model = models.Sequential([
        # 第一个卷积块
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 第二个卷积块
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 第三个卷积块
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # 展平层
        layers.Flatten(),
        
        # 全连接层
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # 输出层
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # 使用更好的优化器配置
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_focal_loss_model(input_shape=(28, 28, 1), num_classes=10, alpha=0.25, gamma=2.0):
    """
    创建使用焦点损失（Focal Loss）的改进模型
    焦点损失专门用于处理难分类样本（如5和6的混淆）
    
    Args:
        input_shape: 输入图像的形状
        num_classes: 分类数量
        alpha: 平衡因子
        gamma: 聚焦参数（越大越关注难分类样本）
    
    Returns:
        model: 编译好的Keras模型
    """
    # 定义焦点损失函数
    def focal_loss(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.where(keras.backend.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = keras.backend.ones_like(y_true) * alpha
        alpha_t = tf.where(keras.backend.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -keras.backend.log(p_t)
        weight = alpha_t * keras.backend.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        return keras.backend.mean(keras.backend.sum(loss, axis=1))
    
    model = models.Sequential([
        # 第一个卷积块
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 第二个卷积块
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 第三个卷积块
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # 展平层
        layers.Flatten(),
        
        # 全连接层
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # 输出层
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # 使用焦点损失编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss,
        metrics=['accuracy']
    )
    
    return model


def create_attention_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    创建带注意力机制的CNN模型
    注意力机制可以帮助模型更好地关注5和6的区别特征
    
    Args:
        input_shape: 输入图像的形状
        num_classes: 分类数量
    
    Returns:
        model: 编译好的Keras模型
    """
    inputs = layers.Input(shape=input_shape)
    
    # 第一个卷积块
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # 第二个卷积块
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # 第三个卷积块
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # 简单的通道注意力
    attention = layers.GlobalAveragePooling2D()(x)
    attention = layers.Dense(128 // 4, activation='relu')(attention)
    attention = layers.Dense(128, activation='sigmoid')(attention)
    attention = layers.Reshape((1, 1, 128))(attention)
    x = layers.Multiply()([x, attention])
    
    x = layers.Dropout(0.25)(x)
    
    # 展平层
    x = layers.Flatten()(x)
    
    # 全连接层
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # 输出层
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
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
    
    print("\n创建改进的CNN模型...")
    improved_cnn_model = create_improved_cnn_model()
    print_model_summary(improved_cnn_model)
    
    print("\n创建全连接模型...")
    dense_model = create_dense_model()
    print_model_summary(dense_model)

