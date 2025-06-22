from NrrdPredictor import PredictorForNRRD
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from predictors.tools import plot_confusion_matrix, plot_roc_curves
from datetime import datetime

def test_nrrd_predictor(test_txt_path: str, save_dir: str = None):
    """
    测试NRRD预测器的性能
    
    Args:
        test_txt_path: 包含NRRD文件路径和标签的文本文件
        save_dir: 结果保存目录，默认为'test_results/nrrd'
    """
    # 创建NRRD预测器
    nrrd_predictor = PredictorForNRRD()
    
    # 设置保存目录
    if save_dir is None:
        save_dir = Path("test_results") / "nrrd" / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取测试集文件
    test_paths = []
    true_labels = []
    try:
        with open(test_txt_path, 'r') as f:
            for line in f:
                nrrd_path, label = line.strip().split('\t')
                test_paths.append(nrrd_path)
                true_labels.append(label)
    except Exception as e:
        raise Exception(f"读取测试集文件失败: {str(e)}")
    
    print(f"加载测试集完成,共 {len(test_paths)} 个NRRD文件")
    
    # 进行预测
    pred_labels = []
    all_probs = []  # 存储每个预测的所有类别概率
    failed_files = []
    
    for i, nrrd_path in enumerate(test_paths, 1):
        try:
            print(f"\n正在预测 ({i}/{len(test_paths)}): {nrrd_path}")
            # 获取预测结果
            result = nrrd_predictor.predict_summary(nrrd_path)
            if not result:
                print(f"预测失败 {nrrd_path}: 空结果")
                failed_files.append(nrrd_path)
                continue
                
            pred_label = result['dominant_category']
            pred_labels.append(pred_label)
            
            # 打印预测结果
            true_label = true_labels[i-1]
            success = "✅" if pred_label == true_label else "❌"
            print(f"实际值: {true_label}, 预测值: {pred_label} {success}")
                
        except Exception as e:
            print(f"预测失败 {nrrd_path}: {str(e)}")
            failed_files.append(nrrd_path)
            continue
    
    if failed_files:
        print(f"\n有 {len(failed_files)} 个文件预测失败")
    
    # 计算总体准确率
    correct = sum(1 for p, t in zip(pred_labels, true_labels) if p == t)
    accuracy = correct / len(true_labels)
    
    # 获取所有唯一标签
    unique_labels = sorted(list(set(true_labels + pred_labels)))
    
    # 计算混淆矩阵
    cm = confusion_matrix(
        true_labels, 
        pred_labels,
        labels=unique_labels
    )
    
    # 生成分类报告
    report = classification_report(true_labels, pred_labels, zero_division=0)
    
    # 准备报告内容
    test_report = "NRRD预测器测试报告:\n\n"
    test_report += f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    test_report += "测试数据信息:\n"
    test_report += f"测试集文件: {test_txt_path}\n"
    test_report += f"总样本数: {len(test_paths)}\n"
    test_report += f"成功预测数: {len(pred_labels)}\n"
    test_report += f"预测失败数: {len(failed_files)}\n\n"
    
    test_report += "预测性能:\n"
    test_report += f"总体准确率: {accuracy:.4f}\n\n"
    
    test_report += "详细分类报告:\n"
    test_report += report
    
    # 如果有预测失败的文件，将它们添加到报告中
    if failed_files:
        test_report += "\n预测失败的文件:\n"
        for file in failed_files:
            test_report += f"{file}\n"
    
    # 计算 Sensitivity, Specificity 和 AUC
    try:
        # 将标签转换为数值
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y_true_idx = [label_to_idx[label] for label in true_labels]
        y_pred_idx = [label_to_idx[label] for label in pred_labels]
        
        # 二进制化标签
        y_true_bin = label_binarize(y_true_idx, classes=range(len(unique_labels)))
        y_pred_bin = label_binarize(y_pred_idx, classes=range(len(unique_labels)))
        
        # 计算每个类别的指标并取平均
        sensitivities = []
        specificities = []
        aucs = []
        
        for i in range(len(unique_labels)):
            # 计算当前类别的TP, FP, TN, FN
            TP = np.sum((y_true_bin[:, i] == 1) & (y_pred_bin[:, i] == 1))
            FP = np.sum((y_true_bin[:, i] == 0) & (y_pred_bin[:, i] == 1))
            TN = np.sum((y_true_bin[:, i] == 0) & (y_pred_bin[:, i] == 0))
            FN = np.sum((y_true_bin[:, i] == 1) & (y_pred_bin[:, i] == 0))
            
            # 计算 Sensitivity (召回率) 和 Specificity
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            
            # 计算当前类别的AUC
            try:
                auc = roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i])
                aucs.append(auc)
            except:
                continue
        
        # 计算平均值
        mean_sensitivity = np.mean(sensitivities)
        mean_specificity = np.mean(specificities)
        mean_auc = np.mean(aucs) if aucs else 0
        
        # 添加到报告中
        test_report += "\n性能指标:\n"
        test_report += f"Sensitivity: {mean_sensitivity:.4f}\n"
        test_report += f"Specificity: {mean_specificity:.4f}\n"
        test_report += f"AUC: {mean_auc:.4f}\n"
        test_report += f"总体准确率: {accuracy:.4f}\n"
        
    except Exception as e:
        print(f"计算性能指标时出错: {str(e)}")
        test_report += "\n性能指标计算失败\n"
    
    # 保存测试报告
    with open(save_dir / "test_report.txt", "w", encoding="utf-8") as f:
        f.write(test_report)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        cm, 
        unique_labels, 
        save_dir / "confusion_matrix.png",
        title='NRRD预测器混淆矩阵'
    )
    
    # 计算并绘制ROC曲线
    try:
        # 将标签转换为数值
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y_true_idx = [label_to_idx[label] for label in true_labels]
        y_pred_idx = [label_to_idx[label] for label in pred_labels]
        
        # 二进制化标签
        y_true_bin = label_binarize(y_true_idx, classes=range(len(unique_labels)))
        y_pred_bin = label_binarize(y_pred_idx, classes=range(len(unique_labels)))
        
        # 绘制ROC曲线
        plot_roc_curves(
            y_true_bin,
            y_pred_bin,
            unique_labels,
            save_dir / "roc_curve.png",
            title='NRRD预测器ROC曲线'
        )
    except Exception as e:
        print(f"绘制ROC曲线时出错: {str(e)}")
    
    return {
        'accuracy': accuracy,
        'predictions': pred_labels,
        'true_labels': true_labels,
        'report': test_report,
        'confusion_matrix': cm,
        'failed_files': failed_files
    }

if __name__ == '__main__':
    # 测试文件路径
    test_txt_path = r"data/nrrd2jpg-test.txt"
    
    print("\n开始NRRD预测器测试...")
    # 运行测试
    results = test_nrrd_predictor(test_txt_path)
    
    # 打印一些基本结果
    print("\n测试完成！")
    print(f"总体准确率: {results['accuracy']:.4f}")
    print(f"详细结果已保存到: test_results/nrrd/")
