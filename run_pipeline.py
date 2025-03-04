import os
import logging
import time
from typing import List, Dict
import argparse
from pathlib import Path
import yaml
import torch
from optimum.quanto import quantize, qint8

class DatasetConfig:
    """数据集配置类"""
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    @property
    def dataset_name(self) -> str:
        return self.config['dataset_name']
    
    @property
    def data_params(self) -> Dict[str, any]:
        return self.config['data_params']
    
    @property
    def default_params(self) -> Dict[str, any]:
        return self.config['default_params']

def setup_logging(log_dir: str = "logs"):
    """设置日志配置"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/pipeline_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

def run_command(command: str, background: bool = False):
    """执行命令的通用函数"""
    if background:
        command += " &"
    logging.info(f"执行命令: {command}")
    return os.system(command)

class ModelPipeline:
    def __init__(self, 
                 config: DatasetConfig,
                 lora_dir: str,
                 embeddings_path: str,
                 version: str,
                 num_workers: int = 6,
                 quantized: bool = False):
        self.config = config
        self.lora_dir = lora_dir
        self.embeddings_path = embeddings_path
        self.version = version
        self.num_workers = num_workers
        self.quantized = quantized
        if self.quantized:
            self.embeddings_path = f"{embeddings_path}_quantized"
            
    def _get_model_suffix(self):
        """获取模型类型后缀"""
        return "_quantized" if self.quantized else ""
    
    def get_embeddings(self, start_layer: int = 1, end_layer: int = 2, text_layer: int = 2):
        """步骤1: 获取embeddings"""
        logging.info(f"开始获取{self.config.dataset_name} embeddings...")
        
        # 根据是否量化设置不同的保存路径
        save_dir = f"{self.embeddings_path}_quantized" if self.quantized else self.embeddings_path
        
        for i in range(start_layer, end_layer):
            # 主模态embeddings
            cmd = (f"python scripts/get_embedding.py "
                   f"--lora_layers {i} "
                   f"--lora_dir {self.lora_dir} "
                   f"--embedding_dir {save_dir}/embeddings_{i}.pth "  # 使用新的保存路径
                   f"--version {self.version} "
                   f"--dataset {self.config.dataset_name} "
                   f"--data_root {self.config.data_params['data_root']}")
            
            if self.quantized:
                cmd += " --quantized"
            
            # 如果有annotation_file则添加
            if 'annotation_file' in self.config.data_params:
                cmd += f" --annotation_file {self.config.data_params['annotation_file']}"
            
            run_command(cmd, background=(i % (self.num_workers-1) != 0))
            
            # LoRA embeddings
            # 根据层数构建lora路径
            lora_path = os.path.join(
                self.lora_dir,
                str(i)
            )
            
            cmd = (f"python scripts/get_embedding.py "
                   f"--lora_layers {i} "
                   f"--lora_dir {lora_path} "  # 使用新的lora路径
                   f"--embedding_dir {save_dir}/embeddings_lora_{i}.pth "  # 使用新的保存路径
                   f"--version {self.version} "
                   f"--use_lora "
                   f"--dataset {self.config.dataset_name} "
                   f"--data_root {self.config.data_params['data_root']}")
            
            if self.quantized:
                cmd += " --quantized"
            
            if 'annotation_file' in self.config.data_params:
                cmd += f" --annotation_file {self.config.data_params['annotation_file']}"
            
            run_command(cmd, background=(i % (self.num_workers-1) != 0))
            
        for i in range(start_layer, text_layer):
            # 文本embeddings
            if 'annotation_file' in self.config.data_params:
                cmd = (f"python scripts/get_text_embedding.py "
                        f"--lora_layers {i} "
                        f"--lora_dir {self.lora_dir} "
                        f"--embedding_dir {save_dir}/text_embeddings_{i}.pth "  # 使用新的保存路径
                        f"--version {self.version} "
                        f"--dataset {self.config.dataset_name} "
                        f"--data_root {self.config.data_params['data_root']} "
                        f"--annotation_file {self.config.data_params['annotation_file']}")
                
                # 添加img_dict_path参数（如果存在）
                if 'img_dict_path' in self.config.data_params:
                    cmd += f" --img_dict_path {self.config.data_params['img_dict_path']}"
                
                if self.quantized:
                    cmd += " --quantized"
                
                run_command(cmd, background=(i % (self.num_workers-1) != 0))

    def test_model(self, start_layer: int = 1, end_layer: int = 33):
        """步骤2: 测试模型性能"""
        logging.info(f"开始测试{self.config.dataset_name}模型性能...")
        
        for i in range(start_layer, end_layer):
            # 测试原始模型
            cmd = (f"python scripts/test_model.py "
                  f"--vision_num_blocks {i} "
                  f"--version {self.version} "
                  f"--lora_dir {self.lora_dir} "
                  f"--embeddings_path {self.embeddings_path}/embeddings_{i}.pth "
                  f"--dataset {self.config.dataset_name} "
                  f"--data_root {self.config.data_params['data_root']} "
                  f"--annotation_file {self.config.data_params['annotation_file']}")
            
            # 添加img_dict_path参数（如果存在）
            if 'img_dict_path' in self.config.data_params:
                cmd += f" --img_dict_path {self.config.data_params['img_dict_path']}"
            
            run_command(cmd, background=(i % (self.num_workers-1) != 0))
            
            lora_path = os.path.join(
                self.lora_dir,
                str(i)
            )
            
            # 测试LoRA模型
            cmd = (f"python scripts/test_model.py "
                  f"--vision_num_blocks {i} "
                  f"--version {self.version} "
                  f"--lora_dir {lora_path} "
                  f"--embeddings_path {self.embeddings_path}/embeddings_lora_{i}.pth "
                  f"--dataset {self.config.dataset_name} "
                  f"--data_root {self.config.data_params['data_root']} "
                  f"--annotation_file {self.config.data_params['annotation_file']} "
                  f"--use_lora")
            
            # 添加img_dict_path参数（如果存在）
            if 'img_dict_path' in self.config.data_params:
                cmd += f" --img_dict_path {self.config.data_params['img_dict_path']}"
            
            run_command(cmd, background=(i % (self.num_workers-1) != 0))

    def find_optimal_layers(self, topk_values: List[int] = None):
        """步骤3: 获取最优层数"""
        if topk_values is None:
            topk_values = self.config.default_params['topk_values']
            
        logging.info(f"开始寻找{self.config.dataset_name}最优层数...")
        
        for i in topk_values:
            # 原始模型
            cmd = (f"python scripts/find_layers.py "
                  f"--S {i} "
                  f"--version {self.version} "
                  f"--dataset {self.config.dataset_name} "
                  f"--data_root {self.config.data_params['data_root']} "
                  f"--annotation_file {self.config.data_params['annotation_file']}")
            
            # 添加img_dict_path参数（如果存在）
            if 'img_dict_path' in self.config.data_params:
                cmd += f" --img_dict_path {self.config.data_params['img_dict_path']}"
            
            run_command(cmd, background=(i % (self.num_workers-1) != 0))
            
            # LoRA模型
            cmd = (f"python scripts/find_layers.py "
                  f"--S {i} "
                  f"--version {self.version} "
                  f"--dataset {self.config.dataset_name} "
                  f"--data_root {self.config.data_params['data_root']} "
                  f"--annotation_file {self.config.data_params['annotation_file']} "
                  f"--use_lora")
            
            # 添加img_dict_path参数（如果存在）
            if 'img_dict_path' in self.config.data_params:
                cmd += f" --img_dict_path {self.config.data_params['img_dict_path']}"
            
            run_command(cmd, background=(i % (self.num_workers-1) != 0))

    def train_predictor(self, topk_values: List[int] = None):
        """步骤4: 训练预测器模型"""
        if topk_values is None:
            topk_values = self.config.default_params['topk_values']
            
        logging.info(f"开始训练{self.config.dataset_name}预测器模型...")
        
        for i in topk_values:
            # 训练原始模型预测器
            cmd = (f"python scripts/train_predictor.py "
                  f"--S {i} "
                  f"--dataset {self.config.dataset_name} "
                  f"--root {self.embeddings_path} "
                  f"--embeddings_file embeddings_{'{i}'}.pth")
            if self.quantized:
                cmd += " --quantized"
            run_command(cmd, background=(i % (self.num_workers-1) != 0))
            
            # 训练LoRA模型预测器
            cmd = (f"python scripts/train_predictor.py "
                  f"--S {i} "
                  f"--dataset {self.config.dataset_name} "
                  f"--root {self.embeddings_path} "
                  f"--embeddings_file embeddings_lora_{'{i}'}.pth "
                  f"--use_lora")
            if self.quantized:
                cmd += " --quantized"
            run_command(cmd, background=(i % (self.num_workers-1) != 0))

    # def predict(self, topk_values: List[int] = None):
    #     """步骤4: 模型预测"""
    #     if topk_values is None:
    #         topk_values = self.config.default_params['topk_values']
            
    #     logging.info(f"开始{self.config.dataset_name}模型预测...")
        
    #     for i in topk_values:
    #         cmd = (f"python scripts/predict.py "
    #               f"--S {i} "
    #               f"--version {self.version} "
    #               f"--root {self.embeddings_path} "
    #               f"--dataset {self.config.dataset_name} "
    #               f"--data_root {self.config.data_params['data_root']} "
    #               f"--annotation_file {self.config.data_params['annotation_file']}")
    #         run_command(cmd, background=(i % (self.num_workers-1) != 0))

    def run_e2e(self,  N: int = 2, Q: int = 100):
        """步骤4: 运行端到端测试"""
    
        logging.info(f"开始{self.config.dataset_name}端到端测试...")
        
        for i in self.config.default_params['topk_values']:
            # 原始模型测试
            cmd = (f"python scripts/run_e2e.py "
                  f"--dataset {self.config.dataset_name} "
                  f"--root {self.embeddings_path} "
                  f"--N {N} "
                  f"--Q {Q} "
                  f"--S {i} ")
                #   f"--device {self.device}")
            if self.quantized:
                cmd += " --quantized"
            run_command(cmd, background=(i % (self.num_workers-1) != 0))
            
            # LoRA模型测试
            cmd = (f"python scripts/run_e2e.py "
                  f"--dataset {self.config.dataset_name} "
                  f"--root {self.embeddings_path} "
                  f"--N 2 "
                  f"--Q 100 "
                  f"--S {i} "
                  # f"--device {self.device} "
                  f"--use_lora")

            
            if self.quantized:
                cmd += " --quantized"
            
            run_command(cmd, background=(i % (self.num_workers-1) != 0))

    def run_pipeline(self, steps: List[bool] = None):
        """运行完整pipeline"""
        if steps is None:
            steps = [True] * 6  # 现在是6个步骤
            
        if steps[0]:
            self.get_embeddings()
        if steps[1]:    
            self.test_model()
        if steps[2]:
            self.find_optimal_layers()
        if steps[3]:
            self.train_predictor()
        # if steps[4]:
        #     self.predict()
        if steps[5]:
            self.run_e2e()

def main():
    parser = argparse.ArgumentParser(description='模型处理Pipeline')
    parser.add_argument('--config', type=str, required=True, help='数据集配置文件路径')
    parser.add_argument('--lora_dir', type=str, help='LoRA模型目录')
    parser.add_argument('--embeddings_path', type=str, help='Embeddings保存路径')
    parser.add_argument('--version', type=str, help='实验版本')
    parser.add_argument('--num_workers', type=int, default=6, help='并行工作进程数')
    parser.add_argument('--steps', type=str, default='111111',
                      help='要运行的步骤(1表示运行,0表示跳过),例如"101111"表示跳过第二步')
    parser.add_argument('--quantized', action='store_true', help='是否使用量化模型')
    
    args = parser.parse_args()
    
    # 加载数据集配置
    config = DatasetConfig(args.config)
    
    # 使用配置文件中的默认值
    lora_dir = args.lora_dir or config.default_params['lora_dir']
    embeddings_path = args.embeddings_path or config.default_params['embeddings_path']
    version = args.version or config.default_params['version']
    
    setup_logging()
    
    # 将steps字符串转换为布尔列表
    steps = [bool(int(x)) for x in args.steps]
    
    # 创建原始pipeline
    pipeline = ModelPipeline(
        config=config,
        lora_dir=lora_dir,
        embeddings_path=embeddings_path,
        version=version,
        num_workers=args.num_workers
    )
    
    if args.quantized:
        # 创建量化pipeline
        quantized_pipeline = ModelPipeline(
            config=config,
            lora_dir=lora_dir,
            embeddings_path=embeddings_path,
            version=f"{version}_quantized",
            num_workers=args.num_workers,
            quantized=True
        )
        # 运行量化pipeline
        quantized_pipeline.run_pipeline(steps)
    else:
        # 运行原始pipeline
        pipeline.run_pipeline(steps)

if __name__ == "__main__":
    main() 