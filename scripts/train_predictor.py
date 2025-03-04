import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import logging
import time
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

class PredictorModel(nn.Module):
    """预测模型"""
    def __init__(self, input_size=1024, hidden_size=256, output_size=32):
        super(PredictorModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//4)
        self.fc3 = nn.Linear(hidden_size//4, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PredictorConfig:
    """预测器配置类"""
    @staticmethod
    def get_config(dataset_name: str):
        configs = {
            'twitter': {
                'input_size': 1024,
                'output_size': 32,
                'batch_size': 4,
                'num_epochs': 50
            },
            'clotho': {
                'input_size': 1024,
                'output_size': 12,
                'batch_size': 4,
                'num_epochs': 50
            },
            'coco': {
                'input_size': 1024,
                'output_size': 32,
                'batch_size': 4,
                'num_epochs': 50
            },
            'flickr8k': {
                'input_size': 1024,
                'output_size': 32,
                'batch_size': 4,
                'num_epochs': 50
            },
            'harsmart': {
                'input_size': 1024,
                'output_size': 6,
                'batch_size': 4,
                'num_epochs': 50
            }
        }
        return configs.get(dataset_name)

def load_data(args, config):
    """加载数据"""
    embeddings_dict = {}
    for i in range(1, config['output_size'] + 1):
        file_path = Path(args.root) / args.embeddings_file.format(i=i)
        embeddings_dict[str(i)] = torch.load(
            file_path, 
            map_location=args.device
        )['vision_embeddings']

    # 加载层数标签
    layers_file = Path(f'results/{args.dataset}/{args.model_type}{args.save_suffix}/R{args.S}/layers.txt')
    layers = np.loadtxt(layers_file)
    layers = np.concatenate([layers for _ in range(config['output_size'])])

    # 合并embeddings
    all_embeddings = []
    source_indicator = []
    for layer_value in range(1, config['output_size'] + 1):
        if str(layer_value) in embeddings_dict:
            embeddings = embeddings_dict[str(layer_value)]
            all_embeddings.append(embeddings)
            source_indicator.extend([layer_value] * len(embeddings))

    embeddings = torch.cat([e.to(args.device) for e in all_embeddings], dim=0)
    return embeddings, layers, np.array(source_indicator)

def train_model(model, data, config, args):
    """训练模型"""
    X_train, y_train = data
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(config['num_epochs']):
        for i in range(0, len(X_train), config['batch_size']):
            batch_x = X_train[i:i + config['batch_size']]
            batch_y = y_train[i:i + config['batch_size']].to(args.device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10000 == 0:
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == batch_y).sum().item() / len(batch_y)
                logging.info(f'Batch [{i//config["batch_size"]}/{len(X_train)//config["batch_size"]}] '
                           f'Loss: {loss.item():.4f} Acc: {accuracy:.4f}')

        logging.info(f'Epoch [{epoch+1}/{config["num_epochs"]}], Loss: {loss.item():.4f}')

def evaluate_model(model, data, source_indicator, config, args):
    """评估模型"""
    X_test, y_test = data
    results = {'accuracy': {}, 'layer_mean': {}}
    
    with torch.no_grad():
        # 总体评估
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        total_accuracy = (predicted == y_test.to(args.device)).sum().item() / len(y_test)
        
        # 分层评估
        for layer in range(1, config['output_size'] + 1):
            mask = source_indicator == layer
            if not any(mask):
                continue
                
            layer_outputs = model(X_test[mask])
            _, layer_predicted = torch.max(layer_outputs, 1)
            layer_targets = y_test[mask].to(args.device)
            
            results['accuracy'][layer] = (layer_predicted == layer_targets).sum().item() / len(layer_targets)
            results['layer_mean'][layer] = torch.mean(layer_predicted.float()).item()
            
            logging.info(f'Layer {layer} - Accuracy: {results["accuracy"][layer]:.4f}, '
                        f'Mean: {results["layer_mean"][layer]:.4f}')
    
    return total_accuracy, results

def save_results(total_accuracy, results, args):
    """保存结果"""
    csv_path = Path(f'results/{args.dataset}/predictor_{args.model_type}{args.save_suffix}.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['S', 'error_bar'] + [f'layer{k}' for k in sorted(results['accuracy'].keys())])
        writer.writerow([args.S, total_accuracy] + [results['accuracy'][k] for k in sorted(results['accuracy'].keys())])
        writer.writerow([args.S, total_accuracy] + [results['layer_mean'][k] for k in sorted(results['layer_mean'].keys())])

def plot_results(results, args):
    """绘制结果图表"""
    layer_keys = sorted(results['accuracy'].keys())
    accuracies = [results['accuracy'][k] for k in layer_keys]
    
    plt.figure(figsize=(10, 6))
    plt.plot(layer_keys, accuracies, 'r-', label='Accuracy')
    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.title(f'Layer-wise Accuracy (S={args.S})')
    plt.legend()
    
    save_path = Path(f'results/{args.dataset}/{args.model_type}{args.save_suffix}/predictor_accuracy.pdf')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="训练预测器模型")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--root", type=str, required=True, help="embeddings根目录")
    parser.add_argument("--embeddings_file", type=str, required=True, help="embeddings文件模板")
    parser.add_argument("--S", type=int, default=10, help="R@S值")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA模型")
    parser.add_argument("--save_suffix", type=str, default="", help="保存后缀")
    parser.add_argument("--quantized", action="store_true", help="是否使用量化模型")
    
    args = parser.parse_args()
    args.model_type = "lora" if args.use_lora else "base"
    if args.quantized:
        args.model_type += "_quantized"
    
    # 设置日志
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = Path("logs/predictor")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/S={args.S}_{args.dataset}_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    
    # 获取配置
    config = PredictorConfig.get_config(args.dataset)
    if config is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # 加载数据
    embeddings, layers, source_indicator = load_data(args, config)
    
    # 划分数据集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, source_train, source_test = train_test_split(
        embeddings, torch.tensor(layers).long(), source_indicator, test_size=0.2, random_state=42
    )
    
    # 初始化模型
    model = PredictorModel(
        input_size=config['input_size'],
        output_size=config['output_size']
    ).to(args.device)
    
    # 训练模型
    train_model(model, (X_train, y_train), config, args)
    
    # 评估模型
    total_accuracy, results = evaluate_model(model, (X_test, y_test), source_test, config, args)
    logging.info(f'Total Accuracy: {total_accuracy:.4f}')
    
    # 保存结果
    save_results(total_accuracy, results, args)
    plot_results(results, args)
    
    # 保存模型
    model_dir = Path(f'models/{args.dataset}/{args.model_type}{args.save_suffix}')
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / f'predictor_S={args.S}.pth')

if __name__ == "__main__":
    main() 