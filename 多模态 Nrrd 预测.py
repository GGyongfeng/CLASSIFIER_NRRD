import config as cfg
from predictors import ModelConfig, Predictor_mutiModel_img,Predictor_sigleModel_img
import os
from pathlib import Path

# 使用示例
if __name__ == '__main__':
    # 模型配置列表
    model_configs = [
        ModelConfig(
            name='ResNet18',
            path="./saved_models/ResNet18-img-01-2-3-(128,128).pth",
            label_dict={'01': 0, '2': 1, '3': 2},
            input_size=(128, 128)
        ),
        ModelConfig(
            name='ResNet18',
            path="./saved_models/ResNet18-img-0-12-3-(128,128).pth",
            label_dict={'0': 0, '12': 1, '3': 2},
            input_size=(128, 128)
        ),
        ModelConfig(
            name='ResNet18',
            path="./saved_models/ResNet18-img-0-1-23-(128,128).pth",
            label_dict={'0': 0, '1': 1, '23': 2},
            input_size=(128, 128)
        ),
        ModelConfig(
            name='ResNet18',
            path="./saved_models/ResNet18-img-4-(128,128).pth",
            label_dict={'0': 0, '1': 1, '2': 2, '3': 3},
            input_size=(128, 128)
        ) 
    ]

    # 创建多模型预测器实例
    predictor = Predictor_mutiModel_img(model_configs)

    # 指定要预测的图片目录
    image_dir = r'C:\Users\gyf15\Desktop\project\data_small\data_original_jpg\3'
    
    # 获取所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png')
    for image_path in Path(image_dir).rglob('*'):
        if image_path.suffix.lower() in image_extensions:
            # 对每张图片进行预测
            result = predictor.predict(str(image_path))
            print(f"图片: {image_path.name}, 预测结果: {result['final_prediction']}")