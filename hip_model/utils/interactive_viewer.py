import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
from batch_process import load_points
from utils import calculate_bilateral_ce_angles, visualize_ce_angles

class CEAngleViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("CE角度测量查看器")
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        # 创建左右分栏
        left_frame = ttk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 图片显示区域
        self.fig = plt.Figure(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 控制面板
        controls = ttk.Frame(right_frame)
        controls.pack(fill=tk.X, padx=5, pady=5)
        
        # 添加控制按钮
        ttk.Button(controls, text="上一张", command=self.prev_image).pack(side=tk.LEFT)
        ttk.Button(controls, text="下一张", command=self.next_image).pack(side=tk.LEFT)
        
        # 添加测量结果显示
        self.result_text = tk.Text(right_frame, height=10, width=40)
        self.result_text.pack(fill=tk.BOTH, expand=True) 

    def load_image(self, index):
        """加载指定索引的图片"""
        img_path = self.image_files[index]
        image = plt.imread(str(img_path))
        points = load_points(Path(img_path))
        return image, points

    def prev_image(self):
        """显示上一张图片"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def next_image(self):
        """显示下一张图片"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.update_display()

    def update_display(self):
        """更新显示内容"""
        image, points = self.load_image(self.current_index)
        left_angle, right_angle = calculate_bilateral_ce_angles(points)
        
        # 更新图像显示
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        visualize_ce_angles(image, points, ax=ax)
        self.canvas.draw()
        
        # 更新文本显示
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, 
            f"图片: {self.image_files[self.current_index].name}\n"
            f"左CE角: {left_angle:.1f}°\n"
            f"右CE角: {right_angle:.1f}°\n") 