import config as cfg
from predictors import ModelConfig, Predictor_mutiModel_img

# 使用示例
if __name__ == '__main__':
    predictor = Predictor_mutiModel_img([
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
    ])

    
    
    # 直接使用配置文件中的路径进行测试
    test_results = predictor.test(cfg.TEST_TXT, cfg.IMAGE_DIR)
    
    # 打印测试报告
    print("\n测试报告:")
    print(test_results['report'])
