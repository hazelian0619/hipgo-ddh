# sharp_angle.py

import cv2
import numpy as np
import json

class SharpAngleAnnotator:
    def __init__(self):
        self.points = []
        self.window_name = 'Sharp Angle Annotator'
        
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 只需要标注耻骨联合点
            self.points.append([x, y])
            # 在图像上显示点
            cv2.circle(self.image, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow(self.window_name, self.image)
            
            # 计算夏普角
            if len(self.points) == 1:  # 已标注耻骨联合点
                self.calculate_sharp_angle()
    
    def calculate_sharp_angle(self):
        """计算夏普角"""
        # 水平参考线
        horizontal = np.array([1, 0])
        
        # 计算左侧夏普角
        left_vector = np.array([
            self.ce_points[2][0] - self.points[0][0],  # 左髋臼外缘x - 耻骨联合x
            self.ce_points[2][1] - self.points[0][1]   # 左髋臼外缘y - 耻骨联合y
        ])
        left_angle = np.degrees(np.arccos(np.dot(horizontal, left_vector) / 
                                        (np.linalg.norm(horizontal) * np.linalg.norm(left_vector))))
        
        # 计算右侧夏普角
        right_vector = np.array([
            self.ce_points[3][0] - self.points[0][0],  # 右髋臼外缘x - 耻骨联合x
            self.ce_points[3][1] - self.points[0][1]   # 右髋臼外缘y - 耻骨联合y
        ])
        right_angle = np.degrees(np.arccos(np.dot(horizontal, right_vector) / 
                                         (np.linalg.norm(horizontal) * np.linalg.norm(right_vector))))
        
        # 显示结果
        print(f"左侧夏普角: {left_angle:.2f}°")
        print(f"右侧夏普角: {right_angle:.2f}°")
        
        # 在图像上显示结果
        cv2.putText(self.image, f"Left Sharp: {left_angle:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(self.image, f"Right Sharp: {right_angle:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(self.window_name, self.image)
        
        # 保存结果
        result = {
            'ce_points': self.ce_points,
            'pubic_point': self.points[0],
            'angles': {
                'left_sharp': float(left_angle),
                'right_sharp': float(right_angle)
            }
        }
        
        with open(f"{self.image_path.split('.')[0]}_sharp.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    def run(self, image_path, ce_points):
        """运行标注器"""
        self.image_path = image_path
        self.ce_points = ce_points  # 使用已有的CE角标注点
        
        # 读取图像
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"无法读取图片: {image_path}")
            return
        
        # 显示已有的CE角点
        for point in self.ce_points:
            cv2.circle(self.image, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)
        
        # 设置窗口和鼠标回调
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        
        print("请点击耻骨联合点位置...")
        
        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键退出
                break
            elif key == ord('r'):  # R键重置
                self.points = []
                self.image = cv2.imread(image_path)
                for point in self.ce_points:
                    cv2.circle(self.image, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)
                cv2.imshow(self.window_name, self.image)
        
        cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":
    # 假设我们已经有了CE角的标注点
    ce_points = [
        [100, 100],  # 左股骨头中心
        [300, 100],  # 右股骨头中心
        [150, 150],  # 左髋臼外缘
        [350, 150]   # 右髋臼外缘
    ]
    
    annotator = SharpAngleAnnotator()
    annotator.run('path_to_your_image.jpg', ce_points)
    