import logging
import torch
import numpy as np
from pathlib import Path
import argparse
from torch.utils.data import DataLoader
import pickle
from pathlib import Path
import sys
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# 数据集导入
from api.twitter import twitter
from api.clotho_text2audio import ClothoTextDataset
from api.coco_text2image import CoCo_t2i_Dataset
from api.flickr import flickr8k
from api.harsmart import HARsmart_relative

class LayerConfig:
    """层数配置类"""
    @staticmethod
    def get_config(dataset_name: str):
        configs = {
            'twitter': {
                'max_layers': 32,
                'file_prefix': 'v{}.txt',
                'default_layer': 31
            },
            'clotho': {
                'max_layers': 12,
                'file_prefix': 'a{}.txt',
                'default_layer': 11
            },
            'coco': {
                'max_layers': 32,
                'file_prefix': 'v{}.txt',
                'default_layer': 31
            },
            'flickr8k': {
                'max_layers': 32,
                'file_prefix': 'v{}.txt',
                'default_layer': 31
            },
            'harsmart': {
                'max_layers': 6,
                'file_prefix': 'i{}.txt',
                'default_layer': 5
            }
        }
        return configs.get(dataset_name)

def transform_matrix(indices, matrix):
    """转换矩阵顺序"""
    if len(indices) != matrix.shape[1]:
        raise ValueError("Length of indices must match matrix columns")
    
    sorted_indices = np.argsort(indices)
    sorted_matrix = matrix[:, sorted_indices]
    sorted_sequence = indices[sorted_indices]
    return sorted_sequence, sorted_matrix

def find_first_nonzero_row(array, column_index, threshold=0):
    """找到第一个非零行"""
    for i, row in enumerate(array):
        if row[column_index] > threshold:
            return i
    return len(array)

def get_dataset_and_targets(dataset_name, data_root, annotation_file, img_dict_path=None, device=None):
    """获取数据集和目标"""
    datasets = {
        'twitter': (twitter, True),
        'clotho': (ClothoTextDataset, False),
        'coco': (CoCo_t2i_Dataset, False),
        'flickr8k': (flickr8k, True),
        'harsmart': (HARsmart_relative, False)
    }
    
    dataset_class, needs_dict = datasets.get(dataset_name, (None, False))
    if dataset_class is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    dataset = dataset_class(
        root_dir=data_root if not isinstance(dataset_class, ClothoTextDataset) else None,
        anne_dir=annotation_file if not isinstance(dataset_class, ClothoTextDataset) else annotation_file,
        split='test' if not isinstance(dataset_class, ClothoTextDataset) else None
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 获取目标列表
    target_list = []
    if needs_dict and img_dict_path:
        with open(img_dict_path, 'rb') as f:
            img_dict = pickle.load(f)
        for _, _, image_names in dataloader:
            targets = [img_dict[name] for name in image_names]
            target_list.extend(targets)
    else:
        for batch in dataloader:
            if len(batch) == 2:  # (x, target)
                _, targets = batch
                target_list.extend(targets.numpy())
    
    return np.array(target_list)

def main():
    parser = argparse.ArgumentParser(description="寻找最优层数")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--data_root", type=str, required=True, help="数据根目录")
    parser.add_argument("--annotation_file", type=str, required=True, help="标注文件路径")
    parser.add_argument("--S", type=int, default=10, help="R@S值")
    parser.add_argument("--version", type=str, required=True, help="版本")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA")
    parser.add_argument("--save_suffix", type=str, default="", help="保存后缀")
    parser.add_argument("--img_dict_path", type=str, required=True, help="图片字典路径")
    
    args = parser.parse_args()
    
    # 获取配置
    config = LayerConfig.get_config(args.dataset)
    if config is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # 加载结果矩阵
    model_type = "lora" if args.use_lora else "base"
    result_dir = Path(f'results/{args.dataset}/{model_type}{args.save_suffix}/R{args.S}')
    
    filenames = [config['file_prefix'].format(i) for i in range(1, config['max_layers']+1)]
    files = [result_dir / filename for filename in filenames]
    matrix = np.vstack([np.loadtxt(f) for f in files])
    
    # 获取目标列表
    target_list = get_dataset_and_targets(
        args.dataset, 
        args.data_root, 
        args.annotation_file,
        args.img_dict_path,
        args.device
    )
    
    # 处理矩阵
    sorted_indices, sorted_matrix = transform_matrix(target_list, matrix)
    
    # 找到最优层数
    optimal_layers = []
    current_item = None
    max_sum = -1
    max_sum_indices = 0
    
    for i, item in enumerate(sorted_indices):
        if current_item != item:
            if current_item is not None:
                optimal_layers.append(max_sum)
                max_sum = -1
            current_item = item
        
        layer = find_first_nonzero_row(sorted_matrix, i)
        if layer > max_sum:
            max_sum = layer
            max_sum_indices = i
            
    optimal_layers.append(max_sum)  # 添加最后一组
    
    # 处理默认值
    optimal_layers = np.array(optimal_layers)
    optimal_layers[optimal_layers >= config['max_layers']] = config['default_layer']
    
    # 保存结果
    output_path = result_dir / 'layers.txt'
    np.savetxt(output_path, optimal_layers, fmt='%d')
    logging.info(f"Average optimal layers: {np.mean(optimal_layers):.2f}")

if __name__ == "__main__":
    main() 
    