import os
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from src.utils import get_date
from .base import Predictor_sigleModel_img, ModelConfig
from .tools import plot_confusion_matrix, plot_roc_point, print_progress_bar, format_time
import time
from datetime import datetime


class Predictor_mutiModel_img:
    """多模型预测器类"""
    
    def __init__(self, model_configs, force_cpu=True):
        """
        初始化多模型预测器
        
        Args:
            model_configs: List[ModelConfig] - 模型配置列表
            force_cpu: bool - 是否强制使用CPU
        """
        self.model_configs = model_configs
        self.force_cpu = force_cpu
        self.predictors = []
        self._load_models()
    
    def _load_models(self):
        """内部方法：加载所有模型"""
        print("\n开始加载模型...")
        
        total = len(self.model_configs)
        loaded_count = 0
        failed_count = 0
        
        for i, model_config in enumerate(self.model_configs, 1):
            try:
                if not isinstance(model_config, ModelConfig):
                    raise TypeError(f"模型配置必须是ModelConfig类型,但得到了{type(model_config)}")
                
                print_progress_bar(
                    i, total,
                    prefix='加载模型:',
                    suffix=f'- {model_config.name}'
                )
                
                predictor = Predictor_sigleModel_img(model_config, force_cpu=self.force_cpu)
                self.predictors.append((model_config, predictor))
                loaded_count += 1
                
            except Exception as e:
                failed_count += 1
                print(f"\n模型 {i} 加载失败: {str(e)}")
                continue
        
        print(f"模型加载完成: 成功 {loaded_count} 个, 失败 {failed_count} 个")

    def _combine_predictions(self, results):
        """
        综合多个模型的预测结果
        
        规则：
        - 如果第一个结果为01,第二个为0,第三个为0,则输出0
        - 如果前三个结果分别为01,12,1,则输出1
        - 如果前三个结果为2,12,23,则输出2
        - 如果前三个结果为3,3,23,则输出3
        - 其他情况则以第四个模型(4分类模型)的结果为准
        """
        if len(results) < 4:
            raise ValueError("需要至少4个模型的预测结果")

        # 获取前三个模型的预测结果
        pred1 = results[0]['prediction']['final_prediction']
        pred2 = results[1]['prediction']['final_prediction']
        pred3 = results[2]['prediction']['final_prediction']
        pred4 = results[3]['prediction']['final_prediction']  # 4分类模型的结果

        # 判断逻辑
        if pred1 == '01' and pred2 == '0' and pred3 == '0':
            final_pred = '0'
        elif pred1 == '01' and pred2 == '12' and pred3 == '1':
            final_pred = '1'
        elif pred1 == '2' and pred2 == '12' and pred3 == '23':
            final_pred = '2'
        elif pred1 == '3' and pred2 == '3' and pred3 == '23':
            final_pred = '3'
        else:
            final_pred = pred4

        return final_pred

    def predict(self, image_path):
        """
        使用所有已加载的模型预测单张图片并综合结果
        """
        results = []
        # print(f"\n预测图片: {image_path}")

        for i, (model_config, predictor) in enumerate(self.predictors, 1):
            try:
                prediction = predictor.predict(image_path)
                result = {
                    'model_info': {
                        'name': model_config.name,
                        'path': model_config.path,
                        'label_dict': model_config.label_dict,
                        'model_index': i
                    },
                    'prediction': prediction
                }
                results.append(result)
                
            except Exception as e:
                print(f"模型 {i} 预测失败: {str(e)}")
                continue

        # 输出每个模型的预测结果
        # print("\n各模型预测结果:")
        # for result in results:
        #     model_name = result['model_info']['name']
        #     pred_class = result['prediction']['final_prediction']
        #     confidence = result['prediction']['confidence']
        #     print(f"{model_name}: 类别 {pred_class} (置信度: {confidence:.4f})")

        # 综合判断最终结果
        try:
            final_prediction = self._combine_predictions(results)
            # print(f"最终预测结果: 类别 {final_prediction}")
        except Exception as e:
            print(f"\n综合预测失败: {str(e)}")
            return results

        return {
            'individual_results': results,
            'final_prediction': final_prediction
        }

    def test(self, test_txt_path: str, image_dir: str):
        """
        测试融合模型在测试集上的性能
        """
        # 记录开始时间
        start_time = time.time()
        start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n开始预测: {start_datetime}")
        
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
        failed_images = []
        total = len(test_paths)
        
        for i, image_path in enumerate(test_paths, 1):
            try:
                result = self.predict(image_path)
                all_preds.append(result['final_prediction'])
                
                print_progress_bar(
                    i, total,
                    prefix='预测进度:',
                    suffix=f'- {os.path.basename(image_path)}'
                )
                
            except Exception as e:
                print(f"\n预测失败 {image_path}: {str(e)}")
                failed_images.append(image_path)
                continue
        
        # 计算总用时
        total_time = time.time() - start_time
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n结束预测: {end_datetime}")
        print(f"总用时: {format_time(total_time)}")
        
        if failed_images:
            print(f"\n有 {len(failed_images)} 张图片预测失败")
        
        # 计算总体准确率
        correct = sum(1 for p, t in zip(all_preds, true_labels) if p == t)
        accuracy = correct / len(true_labels)
        
        # 计算单点的TPR和FPR
        # 将标签和预测结果转换为二进制式（一对多）
        unique_labels = sorted(list(set(true_labels + all_preds)))
        y_true_bin = label_binarize(true_labels, classes=unique_labels)
        y_pred_bin = label_binarize(all_preds, classes=unique_labels)
        
        # 计算所有类别的平均TPR和FPR
        tpr = np.mean([np.sum((y_true_bin[:, i] == 1) & (y_pred_bin[:, i] == 1)) / np.sum(y_true_bin[:, i] == 1) 
                       for i in range(len(unique_labels))])
        fpr = np.mean([np.sum((y_true_bin[:, i] == 0) & (y_pred_bin[:, i] == 1)) / np.sum(y_true_bin[:, i] == 0) 
                       for i in range(len(unique_labels))])
        
        # 生成结果保存路径
        save_dir = Path("test_results") / f"EnsembleTestResult-{get_date()}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 准备报告内容
        report = "融合模型测试报告:\n\n"
        report += f"开始时间: {start_datetime}\n"
        report += f"结束时间: {end_datetime}\n"
        report += f"总用时: {format_time(total_time)}\n\n"
        report += "测试参数信息:\n"
        report += f"test_txt_path={test_txt_path}\n"
        report += f"image_dir={image_dir}\n"
        report += f"测试样本总数: {len(test_paths)}\n"
        report += f"成功预测数: {len(all_preds)}\n"
        report += f"预测失败数: {len(failed_images)}\n\n"
        
        # 添加模型信息
        report += "使用的模型配置:\n"
        for i, (config, _) in enumerate(self.predictors, 1):
            report += f"模型 {i}: {config.name} - {config.path}\n"
            report += f"标签映射: {config.label_dict}\n"
            report += f"输入尺寸: {config.input_size}\n\n"
        
        # 添加分类报告
        report += "分类报告:\n"
        report += classification_report(true_labels, all_preds)
        report += f"\n总体准确率: {accuracy:.4f}"
        
        # 保存失败图片信息
        if failed_images:
            report += "\n\n预测失败的图片:\n"
            for img in failed_images:
                report += f"{img}\n"
        
        # 保存报告
        with open(save_dir / "ensemble_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        # 绘制混淆矩阵
        cm = confusion_matrix(true_labels, all_preds)
        plot_confusion_matrix(
            cm,
            unique_labels,
            save_dir / "confusion_matrix.png",
            title='融合模型混淆矩阵'
        )
        
        # 绘制单点ROC图
        plot_roc_point(
            fpr,
            tpr,
            save_dir / "roc_point.png",
            title='融合模型ROC曲线（单点）'
        )

        return {
            'accuracy': accuracy,
            'predictions': all_preds,
            'true_labels': true_labels,
            'report': report,
            'confusion_matrix': cm,
            'failed_images': failed_images,
            'roc_point': {'fpr': fpr, 'tpr': tpr}
        }
    
    def get_loaded_models_info(self):
        """
        获取已加载模型的信息
        
        Returns:
            List[Dict] - 已加载模型的信息列表
        """
        return [
            {
                'name': config.name,
                'path': config.path,
                'label_dict': config.label_dict
            }
            for config, _ in self.predictors
        ]

    def __len__(self):
        """返回已加载的模型数量"""
        return len(self.predictors)
