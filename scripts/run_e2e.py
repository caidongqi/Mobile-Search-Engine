import logging
import torch
import pickle
from pathlib import Path
import numpy as np
import csv
import argparse
from torch.utils.data import DataLoader
from models.imagebind_model import ModalityType
from models.mlp import PredictorModel

class E2EConfig:
    """端到端测试配置类"""
    @staticmethod
    def get_config(dataset_name: str):
        configs = {
            'twitter': {
                'max_layers': 32,
                'input_size': 1024,
                'K_list': [1, 5, 10],
                'batch_size': 64
            },
            'clotho': {
                'max_layers': 12,
                'input_size': 1024,
                'K_list': [1, 5, 10],
                'batch_size': 64
            },
            'coco': {
                'max_layers': 32,
                'input_size': 1024,
                'K_list': [1, 5, 10],
                'batch_size': 64
            },
            'flickr8k': {
                'max_layers': 32,
                'input_size': 1024,
                'K_list': [1, 5, 10],
                'batch_size': 64
            }
        }
        return configs.get(dataset_name)

def load_embeddings(args, config):
    """加载embeddings"""
    # 加载N层embeddings
    coarse_path = Path(args.root) / f'embeddings_{args.N}_{args.model_type}.pth'
    if not coarse_path.exists():
        raise FileNotFoundError(f"找不到coarse embeddings: {coarse_path}")
    
    coarse_embeddings = torch.load(coarse_path, map_location=args.device)['vision_embeddings']
    
    # 加载完整层embeddings
    fine_path = Path(args.root) / f'embeddings_{config["max_layers"]}_{args.model_type}.pth'
    if not fine_path.exists():
        raise FileNotFoundError(f"找不到fine embeddings: {fine_path}")
        
    fine_embeddings = torch.load(fine_path, map_location=args.device)['vision_embeddings']
    
    return coarse_embeddings, fine_embeddings

def predict_layers(model, embeddings, args):
    """预测每个样本的最优层数"""
    layers = []
    with torch.no_grad():
        for embedding in embeddings:
            layer = model(embedding.to(args.device).float())
            _, layer_idx = torch.max(layer, 0)
            layers.append(layer_idx.item() + 1)
    return np.array(layers)

def get_dynamic_embeddings(layers, args, config):
    """获取动态embeddings"""
    dynamic_embeddings = []
    
    for i, layer in enumerate(layers):
        embedding_path = Path(args.root) / f'embeddings_{layer}_{args.model_type}.pth'
        if not embedding_path.exists():
            logging.warning(f"找不到embedding文件: {embedding_path}")
            continue
            
        embeddings = torch.load(embedding_path, map_location=args.device)['vision_embeddings']
        dynamic_embeddings.append(embeddings[i])
        
    return torch.stack(dynamic_embeddings)

def evaluate_retrieval(embeddings, text_embeddings, target, K_list):
    """评估检索性能"""
    results = {f'K={k}': [] for k in K_list}
    
    match_value = text_embeddings @ embeddings.T
    result = torch.softmax(match_value, dim=-1)
    
    for k in K_list:
        _, topk_indices = torch.topk(result, k=k, dim=-1)
        correct = [int(target[i].item() in topk_indices[i]) for i in range(len(target))]
        results[f'K={k}'] = np.mean(correct)
        
    return results

def main():
    parser = argparse.ArgumentParser(description="端到端测试")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--root", type=str, required=True, help="embeddings根目录")
    parser.add_argument("--N", type=int, default=2, help="初始层数")
    parser.add_argument("--Q", type=int, default=100, help="检索范围")
    parser.add_argument("--S", type=int, default=10, help="R@S值")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA模型")
    parser.add_argument("--save_suffix", type=str, default="", help="保存后缀")
    parser.add_argument("--quantized", action="store_true", help="是否使用量化模型")
    
    args = parser.parse_args()
    args.model_type = "lora" if args.use_lora else "base"
    if args.quantized:
        args.model_type += "_quantized"
    
    # 获取配置
    config = E2EConfig.get_config(args.dataset)
    if config is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
        
    # 加载数据和模型
    coarse_embeddings, fine_embeddings = load_embeddings(args, config)
    
    predictor = PredictorModel(
        input_size=config['input_size'],
        output_size=config['max_layers']
    ).to(args.device)
    
    predictor.load_state_dict(torch.load(
        f'models/{args.dataset}/{args.model_type}{args.save_suffix}/predictor_S={args.S}.pth',
        map_location=args.device
    ))
    
    # 预测层数
    layers = predict_layers(predictor, coarse_embeddings, args)
    mean_layer = np.mean(layers)
    logging.info(f"平均预测层数: {mean_layer:.2f}")
    
    # 获取动态embeddings
    dynamic_embeddings = get_dynamic_embeddings(layers, args, config)
    
    # 加载文本embeddings
    text_embeddings = torch.load(
        Path(args.root) / f'text_embeddings_{args.model_type}.pt',
        map_location=args.device
    )
    
    # 评估结果
    results = evaluate_retrieval(
        dynamic_embeddings,
        text_embeddings,
        torch.arange(len(dynamic_embeddings)).to(args.device),
        config['K_list']
    )
    
    # 保存结果
    csv_path = Path(f'results/{args.dataset}/e2e_{args.model_type}{args.save_suffix}.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['total', args.N, args.Q, args.S, mean_layer] + 
                       [results[f'K={k}'] for k in config['K_list']])

if __name__ == "__main__":
    main() 