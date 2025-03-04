import logging
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import argparse
from pathlib import Path
import sys
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
import data
from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA
# 数据集导入
from api.twitter import twitter
from api.clotho_audio import ClothoDataset
from api.coco_get_image import CoCoDataset
from api.flickr import flickr8k
from api.harsmart import HARsmart_relative
from torch.ao.quantization import quantize_dynamic
from optimum.quanto import quantize, qint8

class DatasetFactory:
    """数据集工厂类"""
    @staticmethod
    def get_dataset(dataset_name: str, data_root: str, annotation_file: str = None, transform=None, split='test'):
        """
        获取数据集实例
        Args:
            dataset_name: 数据集名称
            data_root: 数据根目录
            annotation_file: 标注文件路径(可选)
            transform: 数据转换
            split: 数据集划分
        """
        datasets = {
            'twitter': lambda: twitter(
                root_dir=data_root, 
                anne_dir=annotation_file, 
                split=split
            ) if annotation_file else None,
            'clotho': lambda: ClothoDataset(
                csv_file=annotation_file,
                datadir=data_root,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ) if annotation_file else None,
            'coco': lambda: CoCoDataset(
                datadir=data_root, 
                annFile=annotation_file if annotation_file else None,
                transform=transform
            ),
            'flickr8k': lambda: flickr8k(
                root_dir=data_root, 
                anne_dir=annotation_file, 
                split=split
            ) if annotation_file else None,
            'harsmart': lambda: HARsmart_relative(train=(split=='train'))
        }
        
        dataset_fn = datasets.get(dataset_name)
        if dataset_fn is None:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
            
        dataset = dataset_fn()
        if dataset is None:
            raise ValueError(f"Failed to create dataset {dataset_name}")
            
        return dataset

def get_modality_config(dataset_name: str):
    """获取数据集对应的模态配置"""
    configs = {
        'twitter': {
            'modality': ModalityType.VISION,
            'num_blocks_param': 'vision_num_blocks',
            'batch_size': 1
        },
        'clotho': {
            'modality': ModalityType.AUDIO,
            'num_blocks_param': 'audio_num_blocks',
            'batch_size': 64
        },
        'coco': {
            'modality': ModalityType.VISION,
            'num_blocks_param': 'vision_num_blocks',
            'batch_size': 64
        },
        'flickr8k': {
            'modality': ModalityType.VISION,
            'num_blocks_param': 'vision_num_blocks',
            'batch_size': 64
        },
        'harsmart': {
            'modality': ModalityType.IMU,
            'num_blocks_param': 'imu_num_blocks',
            'batch_size': 256
        }
    }
    return configs.get(dataset_name)

def setup_logging(dataset: str, lora_layers: int, version: str):
    """设置日志"""
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = Path('logs/embeddings')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/{dataset}_{lora_layers}_{version}_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="通用Embedding提取脚本")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--data_root", type=str, required=True, help="数据根目录")
    parser.add_argument("--annotation_file", type=str, default=None, help="标注文件路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--lora_layers", type=int, required=True, help="LoRA层数")
    parser.add_argument("--lora_dir", type=str, required=True, help="LoRA目录")
    parser.add_argument("--embedding_dir", type=str, required=True, help="Embedding保存目录")
    parser.add_argument("--version", type=str, required=True, help="版本")
    parser.add_argument("--use_lora", action='store_true', help="是否使用LoRA")
    parser.add_argument("--split", type=str, default='test', help="数据集分割")
    parser.add_argument("--quantized", action='store_true', help="是否使用量化模型")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.dataset, args.lora_layers, args.version)
    
    # 获取配置
    config = get_modality_config(args.dataset)
    if config is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # 初始化模型
    model_kwargs = {config['num_blocks_param']: args.lora_layers}
    model = imagebind_model.imagebind_huge(pretrained=True, **model_kwargs)


    # 应用LoRA
    if args.use_lora:
        model.modality_trunks.update(
            LoRA.apply_lora_modality_trunks(
                model.modality_trunks, 
                rank=4,
                layer_idxs={config['modality']: [i for i in range(1, args.lora_layers+1)]},
                modality_names=[config['modality']]
            )
        )
        LoRA.load_lora_modality_trunks(
            model.modality_trunks, 
            checkpoint_dir=args.lora_dir, 
            postfix="_last"
        )
    
    # 量化模型
    if args.quantized:
        quantize(model, weights=qint8, activations=qint8)
    
    model = model.to(args.device)
    model.eval()
    
    # 准备数据集
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]) if config['modality'] == ModalityType.VISION else None
    
    try:
        dataset = DatasetFactory.get_dataset(
            args.dataset, 
            args.data_root, 
            args.annotation_file, 
            transform=transform,
            split=args.split
        )
    except ValueError as e:
        logging.error(f"Error creating dataset: {e}")
        return
    
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        drop_last=False,
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    # 提取embeddings
    embeddings = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if config['modality'] == ModalityType.AUDIO:
                inputs = {config['modality']: data.load_and_transform_audio_data(batch[0], device=args.device)}
            else:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(args.device)
                inputs = {config['modality']: x}
            
            current_embeddings = model(inputs)[config['modality']]
            
            if embeddings:
                embeddings[config['modality']] = torch.cat([embeddings[config['modality']], current_embeddings], dim=0)
            else:
                embeddings[config['modality']] = current_embeddings
            
            del current_embeddings
            logging.info(f"Processed batch: {batch_idx}")
    
    # 保存embeddings
    Path(args.embedding_dir).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {f'{config["modality"]}_embeddings': embeddings[config['modality']]},
        args.embedding_dir
    )
    logging.info(f"Embeddings saved to {args.embedding_dir}")

if __name__ == "__main__":
    main() 