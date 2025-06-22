import torch
import torch.nn as nn
import cv2
from pathlib import Path
import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from src.model import ResNet18, MobileNetV2Define, ResNet50
from src.utils import get_date
from .tools import plot_confusion_matrix, plot_roc_curves

class ModelConfig:
    def __init__(self, name, path, label_dict, input_size):
        self.name = name
        self.path = path
        self.label_dict = label_dict
        self.input_size = input_size

class Predictor_sigleModel_img:
    def __init__(self, model_config: ModelConfig, force_cpu: bool = False):
        """初始化图片预测器"""
        # 保存模型配置
        self.model_config = model_config
        
        # 设置设备
        if force_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        # 从配置对象获取参数
        self.input_size = model_config.input_size
        self.labels = model_config.label_dict
        
        # 初始化模型
        available_models = {
            'ResNet18': ResNet18,
            'ResNet50': ResNet50,
            'MobileNetV2': lambda: MobileNetV2Define(len(self.labels))
        }
        
        if model_config.name not in available_models:
            raise ValueError(f"不支持的模型类型: {model_config.name}。可用模型类型: {list(available_models.keys())}")
        
        self.model = available_models[model_config.name]().to(self.device)
        
        # 加载模型权重
        try:
            checkpoint = torch.load(
                model_config.path, 
                map_location=self.device,
                weights_only=True
            )
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        except Exception as e:
            raise Exception(f"模型加载失败: {str(e)}")
            
        self.model.eval()
    
    def preprocess_image(self, image):
        """
        预处理图片数据
        
        Args:
            image: 可以是图片路径(str)或numpy数组(np.ndarray)
        """
        if isinstance(image, str):
            # 如果输入是路径，读取图片
            image = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            # 如果输入是numpy数组，直接使用
            pass
        else:
            raise ValueError("image必须是图片路径或numpy数组")
        
        if len(image.shape) == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) > 3:
            image = image[..., 0:3]
            
        image = cv2.resize(image, self.input_size)
        
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)
        image = image / 255.0
        image = (image - 0.5) / 0.5
        
        return image
        
    def predict(self, image):
        """
        预测图片
        
        Args:
            image: 可以是图片路径(str)或numpy数组(np.ndarray)
        """
        if isinstance(image, str):
            if not Path(image).exists():
                raise FileNotFoundError(f"图片文件不存在: {image}")
            
        try:
            image_tensor = self.preprocess_image(image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
        except Exception as e:
            raise Exception(f"图片处理失败: {str(e)}")
            
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_prob, final_prediction = torch.max(probabilities, dim=1)
            
            # 添加安全检查
            if final_prediction.item() >= len(self.labels):
                print(f"警告：模型预测值 {final_prediction.item()} 超出标签范围 {len(self.labels)}")
                predicted_label = list(self.labels.keys())[0]
            else:
                predicted_label = list(self.labels.keys())[final_prediction.item()]
        
        confidence = predicted_prob.item()
        
        all_probs = probabilities[0].cpu().numpy()
        class_probs = {label: float(prob) for label, prob in zip(self.labels.keys(), all_probs)}
        
        return {
            'final_prediction': predicted_label,
            'confidence': confidence,
            'all_probabilities': class_probs,
            'raw_prediction': final_prediction.item()
        }

    def test(self, test_txt_path: str, image_dir: str):
        """测试单个模型在测试集上的性能"""
        # 从测试集文本文件中读取图片路径和标签
        test_paths = []
        true_labels = []
        
        # 读取测试集文件
        try:
            with open(test_txt_path, 'r') as f:
                for line in f:
                    image_path, label = line.strip().split('\t')
                    full_image_path = os.path.join(image_dir, image_path)
                    test_paths.append(full_image_path)
                    true_labels.append(label)
        except Exception as e:
            raise Exception(f"读取测试集文件失败: {str(e)}")
        
        print(f"加载测试集完成,共 {len(test_paths)} 张图片")
        
        # 进行预测
        all_preds = []
        all_probs = []  # 存储每个预测的所有类别概率
        failed_images = []
        
        for i, image_path in enumerate(test_paths, 1):
            try:
                result = self.predict(image_path)
                all_preds.append(result['final_prediction'])
                all_probs.append(list(result['all_probabilities'].values()))  # 存储所有类别的概率
                if i % 10 == 0:
                    print(f"已完成 {i}/{len(test_paths)} 张图片的预测")
            except Exception as e:
                print(f"预测失败 {image_path}: {str(e)}")
                failed_images.append(image_path)
                continue
        
        if failed_images:
            print(f"\n有 {len(failed_images)} 张图片预测失败")
        
        # 计算总体准确率
        correct = sum(1 for p, t in zip(all_preds, true_labels) if p == t)
        accuracy = correct / len(true_labels)
        
        # 将标签转换为数值形式
        label_to_idx = {label: idx for idx, label in enumerate(sorted(self.labels.keys()))}  # 确保排序
        true_labels_idx = [label_to_idx[label] for label in true_labels]
        pred_labels_idx = [label_to_idx[label] for label in all_preds]
        
        # 计算二进制化的标签
        unique_labels = sorted(list(self.labels.keys()))
        classes = range(len(unique_labels))  # 确保类别从0开始连续
        
        # 使用label_binarize时指定类别范围
        y_true_bin = label_binarize(true_labels_idx, classes=classes)
        y_pred_bin = label_binarize(pred_labels_idx, classes=classes)
        y_prob_bin = np.array(all_probs)  # 所有预测的概率矩阵
        
        # 计算性能指标
        try:
            # 计算敏感度（召回率）和特异度
            if len(unique_labels) == 2:  # 二分类
                # 获取正类的预测概率
                y_score = y_prob_bin[:, 1]
                y_true = y_true_bin[:, 0]
                
                # 计算TP, FN, TN, FP
                TP = np.sum((y_true == 1) & (y_pred_bin[:, 0] == 1))
                FN = np.sum((y_true == 1) & (y_pred_bin[:, 0] == 0))
                TN = np.sum((y_true == 0) & (y_pred_bin[:, 0] == 0))
                FP = np.sum((y_true == 0) & (y_pred_bin[:, 0] == 1))
                
                # 计算指标
                sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                
                # 计算AUC
                auc = roc_auc_score(y_true, y_score)
                
                # 打印调试信息
                print("\n调试信息:")
                print(f"标签分布: {np.bincount(y_true.astype(int))}")
                print(f"预测分布: {np.bincount(y_pred_bin[:, 0].astype(int))}")
                print(f"预测概率范围: [{y_score.min():.4f}, {y_score.max():.4f}]")
                
            else:  # 多分类
                # 计算每个类别的指标并取平均
                sensitivities = []
                specificities = []
                
                for i in range(len(unique_labels)):
                    TP = np.sum((y_true_bin[:, i] == 1) & (y_pred_bin[:, i] == 1))
                    FN = np.sum((y_true_bin[:, i] == 1) & (y_pred_bin[:, i] == 0))
                    TN = np.sum((y_true_bin[:, i] == 0) & (y_pred_bin[:, i] == 0))
                    FP = np.sum((y_true_bin[:, i] == 0) & (y_pred_bin[:, i] == 1))
                    
                    if TP + FN > 0:
                        sensitivities.append(TP / (TP + FN))
                    if TN + FP > 0:
                        specificities.append(TN / (TN + FP))
                
                sensitivity = np.mean(sensitivities) if sensitivities else 0
                specificity = np.mean(specificities) if specificities else 0
                
                # 多分类AUC
                auc = roc_auc_score(y_true_bin, y_prob_bin, multi_class='ovr', average='macro')
                
        except Exception as e:
            print(f"计算性能指标时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
            sensitivity = specificity = auc = 0
        
        # 生成结果保存路径
        save_dir = Path("test_results") / f"SingleModelTestResult-{get_date()}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 准备报告内容
        report = "单模型测试报告:\n\n"
        # 添加模型信息
        report += "模型信息:\n"
        report += f"模型类型: {self.model.__class__.__name__}\n"
        report += f"模型路径: {self.model_config.path}\n"  # 使model_config中的path
        report += f"标签映射: {self.labels}\n"
        report += f"输入尺寸: {self.input_size}\n\n"

        report += f"测试样本总数: {len(test_paths)}\n"
        report += f"成功预测数: {len(all_preds)}\n"
        report += f"预测失败数: {len(failed_images)}\n\n"
        
        
        # 添加分类报告
        report += "分类报告:\n"
        report += classification_report(true_labels, all_preds, zero_division=0)
        report += f"\nSensitivity: {sensitivity:.4f}"
        report += f"\nSpecificity: {specificity:.4f}"
        report += f"\nAUC: {auc:.4f}"
        report += f"\n总体准确率: {accuracy:.4f}"
        
        # 保存失败图片信息
        if failed_images:
            report += "\n\n预测失败的图片:\n"
            for img in failed_images:
                report += f"{img}\n"
        
        # 保存报告
        with open(save_dir / "test_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        # 绘制混淆矩阵
        cm = confusion_matrix(
            true_labels, 
            all_preds,
            labels=sorted(list(self.labels.keys()))  # 确保标签顺序一致
        )
        cm = np.flip(cm, axis=0)  # 翻转行
        cm = np.flip(cm, axis=1)  # 翻转列
        plot_confusion_matrix(
            cm, 
            unique_labels, 
            save_dir / "confusion_matrix.png", 
            title='单模型混淆矩阵'
        )
        
        # 绘制ROC曲线
        plot_roc_curves(
            y_true_bin,
            y_prob_bin,
            unique_labels,
            save_dir / "roc_curve.png",
            title='单模型ROC曲线'
        )
        
        results = {
            'accuracy': accuracy,
            'predictions': all_preds,
            'probabilities': all_probs,
            'true_labels': true_labels,
            'report': report,
            'confusion_matrix': cm,
            'failed_images': failed_images,
            'metrics': {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'auc': auc
            }
        }
        
        return results
    
if __name__ == '__main__':
    model_config = ModelConfig(
        name='ResNet18',
        path="./saved_models/ResNet18-img-4-(128,128).pth",
        label_dict={'0': 0, '1': 1, '2': 2, '3': 3},
        input_size=(128, 128)
    )
    predictor = Predictor_sigleModel_img(model_config)
    result = predictor.test(r'data/all-0123/test.txt', r'dataset/image-all')
    print(result)
