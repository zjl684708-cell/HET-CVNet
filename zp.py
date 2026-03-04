import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import csv
import random
from PIL import Image
import dlib
# 初始化 dlib 的人脸检测器
detector = dlib.get_frontal_face_detector()

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = []
        self.transform = transform

        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # 跳过表头
            for row in csv_reader:
                file_path, label = row[0], int(row[1])
                self.data.append((file_path, label))

    def extract_face(self, image_path):
        image = cv2.imread(image_path)
        if image is not None:
            faces = detector(image)
            if len(faces) > 0:
                face = faces[0]
                height, width, _ = image.shape
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                size_bb = int(max(x2 - x1, y2 - y1) * 1.3)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                x1 = max(int(center_x - size_bb // 2), 0)
                y1 = max(int(center_y - size_bb // 2), 0)
                size_bb = min(width - x1, size_bb)
                size_bb = min(height - y1, size_bb)
                cropped_face = image[y1:y1 + size_bb, x1:x1 + size_bb]

                cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
                return cropped_face_pil
            else:
                return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            height, width, _ = image.shape
            white_background = Image.new('RGB', (width, height), (255, 255, 255))
            return white_background

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]

        try:
            # 提取人脸并转换为PIL Image
            face_image = self.extract_face(file_path)

            # 应用数据增强
            if self.transform:
                face_image = self.transform(face_image)

            return face_image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"加载样本 {file_path} 失败: {str(e)}")
            # 返回占位数据避免中断训练
            return torch.zeros(3, 224, 224), torch.tensor(-1, dtype=torch.long)
