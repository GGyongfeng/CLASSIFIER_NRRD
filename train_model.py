import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from src.data_loader import MyDataset
from src.utils import create_dir, dice_coef, get_date
import config as cfg
from torch.utils.tensorboard import SummaryWriter
from src.model import MobileNetV2Define, MLP, CNN, ResNet18, ResNet50
from pathlib import Path
import time

def train(epochs: int, batch_size: int, patience: int = 10, load_model_path: str = None, 
          interval: int = 10, start_epoch: int = 1, result_name: str = "default"):
    """
    训练模型的函数
    :param epochs: 轮次
    :param batch_size: 批次大小
    :param patience: 早停轮次
    :param load_model_path: 如果需要在原来的参数上继续训练, 指明保存的模型地址
    :param interval: 保存模型的间隔
    :param start_epoch: 开始训练的轮次
    """
    print(f"训练参数信息:\n"
        f"epochs={epochs},\n"
        f"batch_size={batch_size},\n"
        f"patience={patience},\n"
        f"load_model_path={load_model_path},\n"
        f"SIZE={cfg.SIZE},\n"
        f"CUDA_DEVICES={cfg.CUDA_DEVICES},\n"
        f"interval={interval},\n"
        f"start_epoch={start_epoch},\n"
        f"result_name={result_name}\n")
    
    # 存放损失的日志目录
    log_dir = os.path.join(Path(__file__).parent.resolve(), f"logs/logs-{result_name}")
    os.makedirs(log_dir, exist_ok=True)
    print("logs存放目录:", log_dir)

    # 存放模型的目录
    model_dir = os.path.join(Path(__file__).parent.resolve(), f'saved_models/models-{result_name}')
    os.makedirs(model_dir, exist_ok=True)
    print("models存放目录:", model_dir)

    # 创建TensorBoard摘要记录器
    summaryWriter = SummaryWriter(log_dir)

    # 设置设备
    device = torch.device('cuda' if len(cfg.CUDA_DEVICES) > 0 else 'cpu')

    # 数据加载
    train_dataset = MyDataset(cfg.TRAIN_TXT, cfg.IMAGE_DIR, cfg.SIZE, is_train=True)
    validate_dataset = MyDataset(cfg.VALIDATE_TXT, cfg.IMAGE_DIR, cfg.SIZE, is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
    print("数据载入完成...")

    # 初始化模型
    # net = MobileNetV2Define(len(cfg.LABEL_DICT)).to(device)  # 初始化MobileNetV2模型
    # net = MLP().to(device)  # 全连接网络
    # net = CNN().to(device)  # 卷积神经网络
    net = ResNet18().to(device)  # ResNet18模型
    # net = ResNet50().to(device)  # ResNet50模型 

    # 初始化早停相关变量
    best_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    # 加载模型
    if load_model_path is not None:
        checkpoint = torch.load(load_model_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
            best_acc = checkpoint.get('best_acc', 0)
            best_epoch = checkpoint.get('best_epoch', 0)
        else:
            net.load_state_dict(checkpoint)
        print(f'load model: {load_model_path}')
    else:
        print('no model loaded')

    if len(cfg.CUDA_DEVICES) >= 2:
        net = nn.DataParallel(net).to(device)

    optimizer = optim.Adam(net.parameters()) # 定义Adam优化器
    loss_fn = nn.CrossEntropyLoss() # 定义交叉熵损失函数

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        print(f'\nepoch: {epoch}, start time: {get_date()}')

        train_loss = []
        validate_loss = []

        # 训练过程
        net.train()
        for i, (image, label) in enumerate(train_dataloader):
            image = image.to(device)
            label = label.to(device)

            out = net(image)
            loss = loss_fn(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            print(f"epoch:{epoch} -> {i}/{len(train_dataloader)} -> train loss: {loss.item()}")

        # 测试过程
        net.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for j, (image, label) in enumerate(validate_dataloader):
                image = image.to(device)
                label = label.to(device)

                out = net(image)
                loss = loss_fn(out, label)
                validate_loss.append(loss.item())

                total += image.shape[0]
                out = torch.softmax(out, dim=1)
                pre = torch.argmax(out, dim=1)
                correct += torch.sum(torch.eq(pre, label)).item()

        # 计算指标
        average_train_loss = np.array(train_loss).mean()
        average_validate_loss = np.array(validate_loss).mean()
        current_acc = correct / total

        # 更新TensorBoard
        summaryWriter.add_scalars('loss', {
            'train': average_train_loss,
            'validate': average_validate_loss
        }, global_step=epoch)
        summaryWriter.add_scalar('accuracy', current_acc, global_step=epoch)

        # 计算训练用时
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch} Training time length: {epoch_time:.2f}s")
        print(f"average train loss: {average_train_loss}, average validate loss: {average_validate_loss}, "
              f"accuracy: {current_acc}")

        # 保存最佳模型
        if current_acc > best_acc:
            best_acc = current_acc
            best_epoch = epoch
            patience_counter = 0
            
            # 保存最佳模型
            best_model_path = os.path.join(model_dir, f"best-model.pth")
            save_dict = {
                'model_state_dict': net.module.state_dict() if len(cfg.CUDA_DEVICES) >= 2 else net.state_dict(),
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'epoch': epoch
            }
            torch.save(save_dict, best_model_path)
            print(f'bast-model saved: ./saved_models/models-{result_name}/best-model.pth, 准确率: {best_acc}')
        else:
            patience_counter += 1

        # 定期保存检查点
        if epoch % interval == 0:
            checkpoint_path = os.path.join(model_dir, f"net-{result_name}-{epoch}.pth")
            save_dict = {
                'model_state_dict': net.module.state_dict() if len(cfg.CUDA_DEVICES) >= 2 else net.state_dict(),
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'epoch': epoch
            }
            torch.save(save_dict, checkpoint_path)
            print(f'model saved: ./saved_models/models-{result_name}/net-{result_name}-{epoch}.pth')

        # 早停机制
        if patience_counter >= patience:
            print(f"Early stopping triggered! No improvement for {patience} epochs.")
            print(f"Best accuracy: {best_acc} at epoch {best_epoch}")
            break



if __name__ == '__main__':
    # 保存logs和model的文件夹名
    result_name = f"{cfg.MODEL_NAME}-{get_date()}"  
    # result_name = None 

    # 可在现有模型基础上训练
    # load_model_path = "./saved_models/models-None/best-model.pth" 
    load_model_path = None

    if load_model_path and os.path.exists(load_model_path):
        checkpoint = torch.load(load_model_path)
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 1  # 如果从检查点恢复，设置相应的起始轮次
    
    # 训练过程代码...
    train(epochs=10,  # 总轮次
        batch_size=6,
        patience=30,  # 早停轮次
        load_model_path=load_model_path,  # 载入模型
        interval=100,  # 模型保存间隔
        start_epoch=start_epoch,  # 开始轮次
        result_name=result_name  # logs和models的名字
        )