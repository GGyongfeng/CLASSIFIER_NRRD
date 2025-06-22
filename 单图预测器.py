import config as cfg
from predictors import ModelConfig, Predictor_mutiModel_img, Predictor_sigleModel_img
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

class PredictGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("图像预测系统")
        self.root.geometry("800x600")
        
        # 设置默认值
        self.default_image = r"C:\Users\gyf15\Desktop\project\Classifier_JPG\dataset\image-ori\0\723468-33.jpg"
        self.default_model = "./saved_models/models-ResNet18-0-1-23-1121-2345/best-model.pth"
        
        # 创建变量并设置默认值
        self.image_path = tk.StringVar(value=self.default_image)
        self.model_path = tk.StringVar(value=self.default_model)
        self.selected_model = tk.StringVar(value='ResNet18')
        
        self.create_widgets()
        
    def create_widgets(self):
        # 图片路径框
        frame1 = ttk.LabelFrame(self.root, text="图片路径", padding="10")
        frame1.pack(fill="x", padx=10, pady=5)
        
        ttk.Entry(frame1, textvariable=self.image_path, width=50).pack(side="left", padx=5)
        ttk.Button(frame1, text="浏览", command=self.select_image).pack(side="left", padx=5)
        
        # 模型路径框
        frame2 = ttk.LabelFrame(self.root, text="模型路径", padding="10")
        frame2.pack(fill="x", padx=10, pady=5)
        
        ttk.Entry(frame2, textvariable=self.model_path, width=50).pack(side="left", padx=5)
        ttk.Button(frame2, text="浏览", command=self.select_model).pack(side="left", padx=5)
        
        # 模型类型选择
        frame3 = ttk.LabelFrame(self.root, text="模型配置", padding="10")
        frame3.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(frame3, text="模型类型:").pack(side="left", padx=5)
        models = ['ResNet18', 'MobileNetV2Define', 'ResNet50', 'ResNet34']
        model_combo = ttk.Combobox(frame3, textvariable=self.selected_model, values=models)
        model_combo.pack(side="left", padx=5)
        
        # 重置按钮和预测按钮
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="重置为默认值", command=self.reset_to_default).pack(side="left", padx=5)
        ttk.Button(button_frame, text="开始预测", command=self.predict).pack(side="left", padx=5)
        
        # 结果显示区域
        self.result_text = tk.Text(self.root, height=10, width=60)
        self.result_text.pack(pady=10)
        
    def reset_to_default(self):
        self.image_path.set(self.default_image)
        self.model_path.set(self.default_model)
        self.selected_model.set('ResNet18')
        
    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.image_path.set(file_path)
            
    def select_model(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PTH files", "*.pth")]
        )
        if file_path:
            self.model_path.set(file_path)
            
    def predict(self):
        try:
            predictor = Predictor_sigleModel_img(ModelConfig(
                name=self.selected_model.get(),
                path=self.model_path.get(),
                label_dict=cfg.LABEL_DICT,
                input_size=cfg.SIZE
            ))
            
            results = predictor.predict(self.image_path.get())
            
            # 清空之前的结果
            self.result_text.delete(1.0, tk.END)
            # 显示新结果
            self.result_text.insert(tk.END, f"预测结果: {results['final_prediction']}")
            
        except Exception as e:
            messagebox.showerror("错误", f"预测过程出错: {str(e)}")

if __name__ == '__main__':
    root = tk.Tk()
    app = PredictGUI(root)
    root.mainloop()