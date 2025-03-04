import logging
import torch
import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
import data
from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA
from torch.utils.data import DataLoader
import numpy as np
import csv
import time
import argparse
import pickle

# 数据集导入
from api.twitter import twitter
from api.clotho_text2audio import ClothoTextDataset
from api.coco_text2image import CoCo_t2i_Dataset
from api.flickr import flickr8k
from api.harsmart import HARsmart_relative

class TestConfig:
    """测试配置类"""
    @staticmethod
    def get_config(dataset_name: str, img_dict_path: str = None):
        configs = {
            'twitter': {
                'modality': ModalityType.VISION,
                'query_modality': ModalityType.TEXT,
                'batch_size': 64,
                'text_prompt': 'a photo of {}.',
                'img_dict_path': img_dict_path,
                'topk_values': [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            },
            'clotho': {
                'modality': ModalityType.AUDIO,
                'query_modality': ModalityType.TEXT,
                'batch_size': 64,
                'text_prompt': 'a photo of {}.',
                'img_dict_path': img_dict_path,
                'topk_values': [1, 5, 10, 150, 160, 170, 180, 200]
            },
            'coco': {
                'modality': ModalityType.VISION,
                'query_modality': ModalityType.TEXT,
                'batch_size': 64,
                'text_prompt': 'a photo of {}.',
                'img_dict_path': img_dict_path,
                'topk_values': [1, 5, 10, 20, 30, 40, 50, 100, 200]
            },
            'flickr8k': {
                'modality': ModalityType.VISION,
                'query_modality': ModalityType.TEXT,
                'batch_size': 1,
                'text_prompt': 'a photo of {}.',
                'img_dict_path': img_dict_path,
                'topk_values': [1, 5, 10, 20, 30, 40, 50, 100]
            },
            'harsmart': {
                'modality': ModalityType.IMU,
                'query_modality': ModalityType.IMU,
                'batch_size': 256,
                'text_prompt': None,
                'img_dict_path': None,
                'topk_values': [1, 5, 10, 20, 30, 40, 50, 100]
            }
        }
        return configs.get(dataset_name)

class DatasetFactory:
    """数据集工厂类"""
    @staticmethod
    def get_dataset(dataset_name: str, data_root: str, annotation_file: str, split='test'):
        datasets = {
            'twitter': lambda: twitter(
                root_dir=data_root,
                anne_dir=annotation_file,
                split=split
            ),
            'clotho': lambda: ClothoTextDataset(
                csv_file=annotation_file,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ),
            'coco': lambda: CoCo_t2i_Dataset(
                caption_path=annotation_file,
                images_dir=data_root,
                split=split
            ),
            'flickr8k': lambda: flickr8k(
                root_dir=data_root,
                anne_dir=annotation_file,
                split=split
            ),
            'harsmart': lambda: HARsmart_relative(train=False)
        }
        return datasets.get(dataset_name, lambda: None)()

def setup_logging(dataset: str, num_blocks: int, version: str):
    """设置日志"""
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = Path('logs/test')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/{dataset}_{num_blocks}_{version}_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

def evaluate_model(model, dataloader, embeddings, config, device):
    """评估模型性能"""
    counts_rs = {f'counts_r{k}': np.array([]) for k in config['topk_values']}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 处理不同格式的batch数据
            if len(batch) == 3:  # (image, text, image_name)
                _, query_data, image_names = batch
                if config['img_dict_path']:
                    with open(config['img_dict_path'], 'rb') as f:
                        img_dict = pickle.load(f)
                    target = torch.tensor([img_dict[name] for name in image_names]).to(device)
            else:  # (query_data, target)
                query_data, target = batch
                target = target.to(device)
            
            # 准备查询输入
            if config['text_prompt']:
                query_data = [config['text_prompt'].format(x) for x in query_data]
            
            if config['query_modality'] == ModalityType.TEXT:
                inputs = {config['query_modality']: data.load_and_transform_text(query_data, device)}
            elif config['query_modality'] == ModalityType.AUDIO:
                inputs = {config['query_modality']: data.load_and_transform_audio_data(query_data, device)}
            else:
                inputs = {config['query_modality']: query_data.to(device)}
            
            # 计算相似度
            query_embeddings = model(inputs)[config['query_modality']]
            similarity = query_embeddings @ embeddings.T
            similarity = torch.softmax(similarity, dim=-1)
            
            # 计算各个topk的准确率
            _, predicted = torch.max(similarity, -1)
            top_indices_list = [torch.topk(similarity, k=k, dim=-1)[1] for k in config['topk_values']]
            
            for k, top_indices, counts_r in zip(config['topk_values'], top_indices_list, counts_rs.values()):
                if k == 1:
                    counts_rs[f'counts_r{k}'] = np.concatenate([
                        counts_rs[f'counts_r{k}'], 
                        [int(predicted[i] == target[i]) for i in range(len(predicted))]
                    ])
                else:
                    counts_rs[f'counts_r{k}'] = np.concatenate([
                        counts_rs[f'counts_r{k}'], 
                        [int(any(top_indices[i] == target[i])) for i in range(len(target))]
                    ])
            
            # 输出当前batch的性能
            data_length = len(counts_rs['counts_r1'])
            r1 = np.sum(counts_rs['counts_r1']) / data_length
            r5 = np.sum(counts_rs['counts_r5']) / data_length
            r10 = np.sum(counts_rs['counts_r10']) / data_length
            logging.info(f"Batch: {batch_idx}, R@1={r1:.4f}, R@5={r5:.4f}, R@10={r10:.4f}, Total={data_length}")
    
    return counts_rs, (r1, r5, r10)

def main():
    parser = argparse.ArgumentParser(description="通用模型测试脚本")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--data_root", type=str, required=True, help="数据根目录")
    parser.add_argument("--annotation_file", type=str, required=True, help="标注文件路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--vision_num_blocks", type=int, required=True, help="视觉块数")
    parser.add_argument("--version", type=str, required=True, help="版本")
    parser.add_argument("--lora_dir", type=str, required=True, help="LoRA目录")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Embedding路径")
    parser.add_argument("--split", type=str, default='test', help="数据集分割")
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA模型")
    parser.add_argument("--save_suffix", type=str, default="", help="结果文件后缀")
    parser.add_argument("--img_dict_path", type=str, required=True, help="图片字典路径")
    
    args = parser.parse_args()
    
    # 获取数据集配置
    config = TestConfig.get_config(args.dataset, args.img_dict_path)
    if config is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # 设置日志
    setup_logging(args.dataset, args.vision_num_blocks, args.version)
    
    # 初始化模型
    model = imagebind_model.imagebind_huge(pretrained=True)
    if args.use_lora and args.lora_dir:
        load_module(model.modality_heads, module_name="heads",
                   checkpoint_dir=args.lora_dir, device=args.device)
        model_type = "lora"
    else:
        model_type = "base"
    
    model = model.to(args.device)
    model.eval()
    
    # 加载数据集
    dataset = DatasetFactory.get_dataset(
        args.dataset,
        args.data_root,
        args.annotation_file,
        split=args.split
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 加载embeddings
    checkpoint = torch.load(args.embeddings_path, map_location=args.device)
    embeddings = checkpoint[f'{config["modality"]}_embeddings']
    
    # 评估模型
    counts_rs, (r1, r5, r10) = evaluate_model(model, dataloader, embeddings, config, args.device)
    
    # 保存结果
    result_dir = Path(f'results/{args.dataset}/{model_type}{args.save_suffix}')
    for k in config['topk_values']:
        result_path = result_dir / f'R{k}'
        result_path.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            result_path / f'v{args.vision_num_blocks}.txt',
            counts_rs[f'counts_r{k}'],
            fmt='%d'
        )
    
    # 保存CSV结果
    csv_path = Path(f'results/{args.dataset}/test_results_{model_type}{args.save_suffix}.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([args.vision_num_blocks, r1, r5, r10])
    
    logging.info(f"Final Results - R@1: {r1:.4f}, R@5: {r5:.4f}, R@10: {r10:.4f}")

if __name__ == "__main__":
    main() 