from .config import config as cfg
from .predictors import ModelConfig, Predictor_sigleModel_img
from pathlib import Path

if __name__ == '__main__':
    # 1. 配置模型并打印详细信息
    # model_config = ModelConfig(
    #     name='ResNet18',
    #     path='./saved_models/ResNet18-img-4-(128,128).pth',
    #     label_dict=cfg.LABEL_DICT,
    #     input_size=(128, 128)
    # )
    model_config = ModelConfig(
        name='ResNet18',
        path='./saved_models/ResNet18-img-YN-(128,128).pth',
        label_dict={'N': 0, 'Y': 1},
        input_size=(128, 128)
    )
    
    print("\n模型配置:")
    print(f"模型名称: {model_config.name}")
    print(f"模型路径: {model_config.path}")
    print(f"标签字典: {model_config.label_dict}")
    print(f"输入尺寸: {model_config.input_size}")
    
    predictor = Predictor_sigleModel_img(model_config)
    
    # 2. 测试完整数据集
    # test_file = r'data/all-0123/test.txt'
    test_file = r'data/all-YN/test.txt'
    try:
        result = predictor.test(test_file, cfg.IMAGE_DIR)
        
        print("\n测试报告:")
        print(result['report'])
        
        print("\n性能指标:")
        print(f"准确率: {result['accuracy']:.4f}")
        print(f"敏感度: {result['metrics']['sensitivity']:.4f}")
        print(f"特异度: {result['metrics']['specificity']:.4f}")
        print(f"AUC: {result['metrics']['auc']:.4f}")
        
        if result['failed_images']:
            print(f"\n预测失败的图片数量: {len(result['failed_images'])}")
            print("失败图片列表:")
            for img in result['failed_images'][:5]:  # 只显示前5个
                print(f"- {img}")
            if len(result['failed_images']) > 5:
                print(f"... 还有 {len(result['failed_images'])-5} 个")
    
    except Exception as e:
        print(f"测试过程出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
    