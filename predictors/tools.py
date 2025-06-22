import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
import SimpleITK as sitk
import cv2

def print_progress_bar(current, total, prefix='', suffix='', length=50):
    """
    打印进度条
    
    Args:
        current: int - 当前进度
        total: int - 总数
        prefix: str - 进度条前缀
        suffix: str - 进度条后缀
        length: int - 进度条长度
    
    Returns:
        str - 完整的进度条字符串
    """
    progress = current / total
    filled_length = int(length * progress)
    bar = '=' * filled_length + '-' * (length - filled_length)
    progress_str = f'\r{prefix}[{bar}] {current}/{total} ({progress:.1%}) {suffix}'
    print(progress_str, end='')
    
    # 如果完成了，打印换行
    if current == total:
        print()
    
    return progress_str

def format_time(seconds):
    """
    将秒数格式化为时:分:秒格式
    
    Args:
        seconds: float - 秒数
    
    Returns:
        str - 格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def plot_confusion_matrix(cm, labels, save_path: Path, title='混淆矩阵'):
    """绘制并保存混淆矩阵图"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(10, 8))
    # 翻转标签顺序以匹配矩阵
    labels = list(reversed(labels))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.title(title)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_roc_curves(y_true_bin, y_prob_bin, class_names, save_path, title='ROC Curves'):
    """绘制ROC曲线"""
    plt.figure(figsize=(10, 8))
    
    if len(class_names) == 2:
        # 二分类情况：使用第二个类别的预测概率（对应于正类'Y'）
        fpr, tpr, _ = roc_curve(y_true_bin[:, 0], y_prob_bin[:, 1])  # 使用第二个概率
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[1]} (AUC = {roc_auc:.2f})')  # 使用第二个标签名
    else:
        # 多分类情况：为每个类别绘制ROC曲线
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob_bin[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()

def plot_roc_point(fpr, tpr, save_path: Path, title='ROC曲线（单点）'):
    """
    绘制并保存单点ROC图
    
    Args:
        fpr: float - 假阳性率
        tpr: float - 真阳性率
        save_path: Path - 保存路径
        title: str - 图表标题
    """
    # 设置全局中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(10, 8))
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    # 绘制模型性能点
    plt.plot(fpr, tpr, 'ro', markersize=10, label=f'模型性能 (TPR={tpr:.3f}, FPR={fpr:.3f})')
    # 连接原点和性能点,以及性能点和(1,1)
    plt.plot([0, fpr], [0, tpr], 'r--', alpha=0.5)
    plt.plot([fpr, 1], [tpr, 1], 'r--', alpha=0.5)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title(title)
    plt.legend(loc='lower right', prop={'family': 'SimHei'})
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def nrrd2images(file_path: str) -> tuple[list[np.ndarray], bool]:
    """
    将NRRD文件转换为图片数组
    
    Args:
        file_path: NRRD文件路径
    
    Returns:
        tuple: (图片数组列表, 是否成功)
        - 如果文件不是NRRD或尺寸不是512*512，返回 ([], False)
        - 成功时返回 (图片列表, True)
    """
    try:
        # 检查文件是否为nrrd
        if not file_path.lower().endswith('.nrrd'):
            return [], False
            
        # 读取NRRD文件
        img = sitk.ReadImage(file_path)
        img_array = sitk.GetArrayFromImage(img)
        
        # 检查每一层的尺寸是否为512*512
        if img_array.shape[1] != 512 or img_array.shape[2] != 512:
            return [], False
            
        # 转换每一层为图片
        images = []
        for i in range(img_array.shape[0]):
            # 获取单层数据
            slice_data = img_array[i]
            
            # 归一化到0-255
            slice_data = ((slice_data - slice_data.min()) / 
                         (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
            
            # 确保是3通道图像
            if len(slice_data.shape) == 2:
                slice_data = cv2.cvtColor(slice_data, cv2.COLOR_GRAY2BGR)
                
            images.append(slice_data)
            
        return images, True
        
    except Exception as e:
        print(f"处理NRRD文件时出错: {str(e)}")
        return [], False
