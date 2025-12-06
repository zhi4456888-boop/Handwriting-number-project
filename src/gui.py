"""
图形用户界面
用于手写数字识别测试
"""

import tkinter as tk
from tkinter import Canvas, Button, Label, messagebox
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw, ImageTk
import os
import io


class HandwritingRecognitionGUI:
    def __init__(self, root, model_path=None):
        """
        初始化GUI
        
        Args:
            root: Tkinter根窗口
            model_path: 模型文件路径
        """
        self.root = root
        self.root.title("手写数字识别系统")
        self.root.geometry("600x500")
        
        # 画布尺寸
        self.canvas_width = 280
        self.canvas_height = 280
        
        # 创建画布
        self.canvas = Canvas(
            root, 
            width=self.canvas_width, 
            height=self.canvas_height,
            bg='white',
            cursor='pencil'
        )
        self.canvas.pack(pady=10)
        
        # 绑定鼠标事件
        self.canvas.bind('<Button-1>', self.start_paint)  # 鼠标按下
        self.canvas.bind('<B1-Motion>', self.paint)       # 鼠标拖动
        self.canvas.bind('<ButtonRelease-1>', self.stop_paint)  # 鼠标抬起
        
        # 存储绘制点
        self.last_x = None
        self.last_y = None
        self.is_drawing = False  # 是否正在绘制
        
        # 创建按钮框架
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        
        # 识别按钮
        self.predict_btn = Button(
            button_frame, 
            text="识别数字", 
            command=self.predict_digit,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 12, 'bold'),
            width=12,
            height=2
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # 清除按钮
        self.clear_btn = Button(
            button_frame, 
            text="清除画布", 
            command=self.clear_canvas,
            bg='#f44336',
            fg='white',
            font=('Arial', 12, 'bold'),
            width=12,
            height=2
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # 结果标签
        self.result_label = Label(
            root,
            text="请在画布上绘制数字 (0-9)",
            font=('Arial', 16, 'bold'),
            fg='#333333'
        )
        self.result_label.pack(pady=10)
        
        # 置信度标签
        self.confidence_label = Label(
            root,
            text="",
            font=('Arial', 12),
            fg='#666666'
        )
        self.confidence_label.pack()
        
        # 加载模型
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # 尝试加载默认模型（按优先级顺序）
            default_paths = [
                # 改进模型（优先）
                '../models/mnist_improved_best.h5',
                '../models/mnist_improved_final.h5',
                # 标准CNN模型
                '../models/mnist_cnn_best.h5',
                '../models/mnist_cnn_final.h5',
                '../models/mnist_improved_cnn_best.h5',
                '../models/mnist_improved_cnn_final.h5',
                # 从项目根目录查找
                'models/mnist_improved_best.h5',
                'models/mnist_improved_final.h5',
                'models/mnist_cnn_best.h5',
                'models/mnist_cnn_final.h5',
                # 从src目录查找
                './models/mnist_improved_best.h5',
                './models/mnist_improved_final.h5',
                './models/mnist_cnn_best.h5',
                './models/mnist_cnn_final.h5',
            ]
            
            # 如果以上路径都不存在，尝试查找models目录下的任何.h5文件
            found_model = False
            for path in default_paths:
                if os.path.exists(path):
                    self.load_model(path)
                    found_model = True
                    break
            
            # 如果还没找到，尝试自动搜索models目录
            if not found_model:
                model_dirs = ['../models', 'models', './models']
                for model_dir in model_dirs:
                    if os.path.exists(model_dir):
                        # 查找所有.h5文件
                        for file in os.listdir(model_dir):
                            if file.endswith('.h5') and ('best' in file or 'final' in file):
                                model_path = os.path.join(model_dir, file)
                                try:
                                    self.load_model(model_path)
                                    found_model = True
                                    break
                                except:
                                    continue
                    if found_model:
                        break
            
            if not found_model:
                messagebox.showwarning(
                    "警告",
                    "未找到训练好的模型！\n\n请先运行以下命令之一训练模型：\n"
                    "- train.py (标准模型)\n"
                    "- train_improved.py (改进模型)\n\n"
                    "或者手动指定模型路径：\n"
                    "python gui.py <模型路径>"
                )
    
    def load_model(self, model_path):
        """加载模型"""
        try:
            print(f"正在加载模型: {model_path}")
            self.model = keras.models.load_model(model_path)
            print("模型加载成功!")
            self.result_label.config(text="模型已加载，可以开始识别")
        except Exception as e:
            print(f"加载模型失败: {e}")
            messagebox.showerror("错误", f"无法加载模型: {e}")
    
    def start_paint(self, event):
        """开始绘制（鼠标按下）"""
        self.last_x = event.x
        self.last_y = event.y
        self.is_drawing = True
    
    def paint(self, event):
        """绘制函数（鼠标拖动）"""
        if self.is_drawing and self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=15, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE
            )
        self.last_x = event.x
        self.last_y = event.y
    
    def stop_paint(self, event):
        """停止绘制（鼠标抬起）"""
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        """清除画布"""
        self.canvas.delete("all")
        self.last_x = None
        self.last_y = None
        self.is_drawing = False
        self.result_label.config(text="画布已清除，请重新绘制")
        self.confidence_label.config(text="")
    
    def get_canvas_image(self):
        """获取画布图像并转换为模型输入格式（使用postscript方法）"""
        # 获取画布内容
        self.canvas.update()
        
        try:
            # 将画布转换为PIL图像
            ps = self.canvas.postscript(colormode='mono')
            img = Image.open(io.BytesIO(ps.encode('utf-8')))
            
            # 调整大小为28x28
            img = img.resize((28, 28), Image.LANCZOS)
            
            # 转换为numpy数组
            img_array = np.array(img.convert('L'))
            
            # 反转颜色（画布是白底黑字，需要转换为黑底白字）
            img_array = 255 - img_array
            
            # 归一化
            img_array = img_array.astype('float32') / 255.0
            
            # 调整形状为 (1, 28, 28, 1)
            img_array = img_array.reshape(1, 28, 28, 1)
            
            return img_array
        except Exception as e:
            print(f"使用postscript方法失败: {e}")
            return self.get_canvas_image_alternative()
    
    def get_canvas_image_alternative(self):
        """替代方法：直接从画布获取图像"""
        # 获取画布内容
        self.canvas.update()
        
        # 创建PIL图像
        img = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # 获取画布上的所有对象
        items = self.canvas.find_all()
        for item in items:
            coords = self.canvas.coords(item)
            if len(coords) >= 4:
                # 绘制线条
                for i in range(0, len(coords) - 2, 2):
                    draw.line(
                        (coords[i], coords[i+1], coords[i+2], coords[i+3]),
                        fill='black',
                        width=15
                    )
        
        # 调整大小为28x28
        img = img.resize((28, 28), Image.LANCZOS)
        
        # 转换为numpy数组
        img_array = np.array(img)
        
        # 反转颜色
        img_array = 255 - img_array
        
        # 归一化
        img_array = img_array.astype('float32') / 255.0
        
        # 调整形状
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    
    def predict_digit(self):
        """预测数字"""
        if self.model is None:
            messagebox.showwarning("警告", "模型未加载！")
            return
        
        try:
            # 获取画布图像
            img_array = self.get_canvas_image_alternative()
            
            # 进行预测
            predictions = self.model.predict(img_array, verbose=0)
            predicted_digit = np.argmax(predictions[0])
            confidence = predictions[0][predicted_digit]
            
            # 更新结果标签
            self.result_label.config(
                text=f"识别结果: {predicted_digit}",
                fg='#4CAF50'
            )
            self.confidence_label.config(
                text=f"置信度: {confidence:.2%}"
            )
            
        except Exception as e:
            messagebox.showerror("错误", f"预测失败: {e}")


def main():
    """主函数"""
    import sys
    
    root = tk.Tk()
    
    # 如果提供了模型路径，使用它
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    app = HandwritingRecognitionGUI(root, model_path)
    root.mainloop()


if __name__ == "__main__":
    import io
    main()

