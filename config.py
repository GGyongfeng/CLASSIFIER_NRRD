#!usr/bin/env python3
# -*- coding: UTF-8 -*-

# 图片存放目录  不要有中文
IMAGE_DIR = r'dataset/image-all'

# 训练集的文本路径
TRAIN_TXT = r'data/all-0123/train.txt'

# 验证集的文本路径
VALIDATE_TXT = r'data/all-0123/validate.txt'

# 测试集的文本路径
TEST_TXT = r'data/all-0123/test.txt'

# 分类
LABEL_DICT = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3
}
# LABEL_DICT = {
#         '0': 0,
#         '1': 1,
#         '23': 2
# }

# 储存训练、测试数据、模型参数的文件夹名称
MODEL_NAME = "ResNet18-0-1-23"

# ######### 传入模型图片的宽和高, 通常宽高相等#########
# SIZE = (32, 32)
# SIZE = (64, 64)
SIZE = (128, 128)
# SIZE = (256, 256)
# SIZE = (512, 512)

# ######### 选择训练的GPU型号 #########
# 如果没有GPU, 设置  CUDA_DEVICES = ""

# 假设电脑有3张显卡，且编号分别是 GPU0， GPU1， GPU2
# 如果使用三张卡训练，可设置 CUDA_DEVICES = "0, 1, 2"

CUDA_DEVICES = ""
