from predictors import ModelConfig, Predictor_sigleModel_img, Predictor_mutiModel_img
from predictors.tools import nrrd2images
from typing import List, Dict, Union
import numpy as np
from pathlib import Path
import cv2

class PredictorForNRRD:
    def __init__(self):
        """初始化NRRD预测器，包含默认的二分类和多分类模型配置"""
        # 二分类模型配置
        self.model_config_2 = ModelConfig(
            name='ResNet18',
            path='./saved_models/ResNet18-img-YN-(128,128).pth',
            label_dict={'无特征': 0, '有特征': 1},
            input_size=(128, 128)
        )

        # 多分类模型配置
        self.model_configs_4 = [
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

        # 初始化预测器
        self.yn_predictor = Predictor_sigleModel_img(self.model_config_2)
        self.mult_predictor = Predictor_mutiModel_img(self.model_configs_4)
        
    def predict(self, path: Union[str, Path]) -> List[Dict]:
        """
        预测NRRD文件或图片文件夹中的切片
        
        Args:
            path: NRRD文件路径或包含切片图片的文件夹路径
            
        Returns:
            List[Dict]: 每一层的预测结果列表
            每个字典包含:
            {
                'slice_index': int,  # 切片索引
                'has_feature': bool,  # 是否有特征
                'yn_confidence': float,  # 二分类的置信度
                'category': str,      # 如果有特征，返回具体类别(0-3)，否则为'无特征'
                'category_confidence': float   # 多分类的置信度（如果有）
            }
        """
        path = Path(path)
        
        # 判断输入是NRRD文件还是文件夹
        if path.is_file() and path.suffix.lower() == '.nrrd':
            # 处理NRRD文件
            images, success = nrrd2images(str(path))
            if not success:
                print(f"NRRD文件处理失败: {path}")
                return []
            # 对NRRD文件的每一层进行旋转
            images = [cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) for img in images]
        elif path.is_dir():
            # 处理图片文件夹
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            images = []
            # 获取所有图片文件并按名称排序
            image_files = sorted(
                [f for f in path.iterdir() if f.suffix.lower() in image_extensions],
                key=lambda x: int(''.join(filter(str.isdigit, x.stem))) if any(c.isdigit() for c in x.stem) else x.stem
            )
            
            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"无法读取图片: {img_path}")
                        continue
                    images.append(img)
                except Exception as e:
                    print(f"处理图片时出错 {img_path}: {str(e)}")
                    continue
                    
            if not images:
                print(f"文件夹中没有有效的图片: {path}")
                return []
        else:
            print(f"无效的输入路径: {path}")
            return []
            
        results = []
        
        # 处理每一层
        for i, img in enumerate(images):
            # 首先进行二分类判断
            yn_result = self.yn_predictor.predict(img)
            
            result_dict = {
                'slice_index': i,
                'has_feature': yn_result['final_prediction'] == '有特征',
                'yn_confidence': yn_result['confidence']
            }
            
            # 如果有特征，进行多分类判断
            if result_dict['has_feature']:
                mult_result = self.mult_predictor.predict(img)
                result_dict['category'] = mult_result['final_prediction']
                result_dict['category_confidence'] = mult_result.get('confidence', 0.0)
            else:
                result_dict['category'] = '无特征'
                result_dict['category_confidence'] = result_dict['yn_confidence']
            
            results.append(result_dict)
            
        return results
    
    def predict_summary(self, path: Union[str, Path]) -> Dict:
        """
        生成NRRD文件或图片文件夹的预测总结
        
        Args:
            path: NRRD文件路径或包含切片图片的文件夹路径
            
        Returns:
            Dict: 包含以下信息：
            {
                'total_slices': int,          # 总切片数
                'feature_slices': int,        # 有特征的切片数
                'category_counts': Dict,       # 各类别的计数
                'dominant_category': str,      # 主要类别
                'detailed_results': List[Dict] # 详细的逐层结果
            }
        """
        detailed_results = self.predict(path)
        
        if not detailed_results:
            return {}
            
        # 统计信息
        total_slices = len(detailed_results)
        feature_slices = sum(1 for r in detailed_results if r['has_feature'])
        
        # 统计各类别计数
        category_counts = {}
        for result in detailed_results:
            cat = result['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
        # 确定主要类别（排除"无特征"）
        category_counts_features = {k: v for k, v in category_counts.items() if k != '无特征'}
        dominant_category = max(category_counts_features.items(), key=lambda x: x[1])[0] if category_counts_features else '无特征'
        
        return {
            'total_slices': total_slices,
            'feature_slices': feature_slices,
            'category_counts': category_counts,
            'dominant_category': dominant_category,
            'detailed_results': detailed_results
        }

    def save_nrrd_slices(self, nrrd_path: Union[str, Path], output_dir: Union[str, Path]) -> bool:
        """
        将NRRD文件转换为图片并保存到指定目录，图片会逆时针旋转90度以匹配训练数据方向
        
        Args:
            nrrd_path: NRRD文件路径
            output_dir: 输出目录路径
            
        Returns:
            bool: 转换是否成功
        """
        nrrd_path = Path(nrrd_path)
        output_dir = Path(output_dir)
        
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 转换NRRD为图片数组
        images, success = nrrd2images(str(nrrd_path))
        
        if not success:
            print(f"NRRD文件处理失败: {nrrd_path}")
            return False
        
        # 保存每一层图片
        for i, img in enumerate(images):
            # 逆时针旋转90度
            rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            output_path = output_dir / f"slice_{i:03d}.jpg"
            try:
                cv2.imwrite(str(output_path), rotated_img)
                print(f"已保存: {output_path}")
            except Exception as e:
                print(f"保存图片失败 {output_path}: {str(e)}")
                return False
            
        return True

if __name__ == '__main__':
    # 创建NRRD预测器
    nrrd_predictor = PredictorForNRRD()
    
    # NRRD文件路径
    nrrd_path = r'C:\Users\gyf15\Desktop\project\data_small\NRRD\2\592305\image.nrrd'
    # 转换后的图片保存路径
    output_dir = r'C:\Users\gyf15\Desktop\project\data_small\NRRD_converted\2\592305\image'
    
    # 保存NRRD切片
    print("\n保存NRRD切片:")
    nrrd_predictor.save_nrrd_slices(nrrd_path, output_dir)
    
    print("\n比较两种方式的预测结果:")
    print("\n1. 直接预测NRRD:")
    nrrd_results = nrrd_predictor.predict_summary(nrrd_path)
    print("NRRD预测结果:")
    print(f"总切片数: {nrrd_results['total_slices']}")
    print(f"有特征切片数: {nrrd_results['feature_slices']}")
    print("各类别计数:", nrrd_results['category_counts'])
    print(f"主要类别: {nrrd_results['dominant_category']}")
    
    print("\n2. 预测已有JPG:")
    jpg_path = r'C:\Users\gyf15\Desktop\project\data_small\JPG\2\592305\image'
    jpg_results = nrrd_predictor.predict_summary(jpg_path)
    print("JPG预测结果:")
    print(f"总切片数: {jpg_results['total_slices']}")
    print(f"有特征切片数: {jpg_results['feature_slices']}")
    print("各类别计数:", jpg_results['category_counts'])
    print(f"主要类别: {jpg_results['dominant_category']}")
    
    print("\n3. 预测转换后的图片:")
    converted_results = nrrd_predictor.predict_summary(output_dir)
    print("转换后图片预测结果:")
    print(f"总切片数: {converted_results['total_slices']}")
    print(f"有特征切片数: {converted_results['feature_slices']}")
    print("各类别计数:", converted_results['category_counts'])
    print(f"主要类别: {converted_results['dominant_category']}") 