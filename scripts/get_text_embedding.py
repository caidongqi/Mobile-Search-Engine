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
import time
import argparse
from pathlib import Path
import pickle
from optimum.quanto import quantize, qint8


# 数据集导入
from api.twitter import twitter
from api.clotho_text2audio import ClothoTextDataset
from api.coco_text2image import CoCo_t2i_Dataset
from api.flickr import flickr8k

class TextDatasetFactory:
    """文本数据集工厂类"""
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
            )
        }
        return datasets.get(dataset_name, lambda: None)()

def get_text_config(dataset_name: str, img_dict_path: str = None):
    """获取数据集文本配置"""
    configs = {
        'twitter': {
            'batch_size': 1,
            'text_prompt': 'a photo of {}.',
            'img_dict_path': img_dict_path
        },
        'clotho': {
            'batch_size': 64,
            'text_prompt': 'a photo of {}.',
            'img_dict_path': img_dict_path
        },
        'coco': {
            'batch_size': 64,
            'text_prompt': 'a photo of {}.',
            'img_dict_path': img_dict_path
        },
        'flickr8k': {
            'batch_size': 64,
            'text_prompt': 'a photo of {}.',
            'img_dict_path': img_dict_path
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
    parser = argparse.ArgumentParser(description="通用文本Embedding提取脚本")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--data_root", type=str, required=True, help="数据根目录")
    parser.add_argument("--annotation_file", type=str, required=True, help="标注文件路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--lora_layers", type=int, required=True, help="LoRA层数")
    parser.add_argument("--lora_dir", type=str, required=True, help="LoRA目录")
    parser.add_argument("--embedding_dir", type=str, required=True, help="Embedding保存目录")
    parser.add_argument("--version", type=str, required=True, help="版本")
    parser.add_argument("--use_lora", action='store_true', help="是否使用LoRA")
    parser.add_argument("--split", type=str, default='test', help="数据集分割")
    parser.add_argument("--img_dict_path", type=str, required=True, help="图片字典路径")
    parser.add_argument("--quantized", action='store_true', help="是否使用量化模型")

    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.dataset, args.lora_layers, args.version)
    
    # 根据是否量化设置不同的保存路径
    if args.quantized:
        embedding_dir = Path(args.embedding_dir)
        args.embedding_dir = str(embedding_dir.parent / f"{embedding_dir.name}_quantized")
    
    # 获取数据集配置，传入img_dict_path参数
    config = get_text_config(args.dataset, args.img_dict_path)
    if config is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # 初始化模型
    model = imagebind_model.imagebind_huge(pretrained=True)
    
    # 应用LoRA
    if args.use_lora:
        model.modality_trunks.update(
            LoRA.apply_lora_modality_trunks(
                model.modality_trunks, 
                rank=4,
                layer_idxs={ModalityType.TEXT: [i for i in range(1, args.lora_layers+1)]},
                modality_names=[ModalityType.TEXT]
            )
        )
        LoRA.load_lora_modality_trunks(
            model.modality_trunks, 
            checkpoint_dir=args.lora_dir, 
            postfix="_last"
        )
    
    # 量化模型
    if args.quantized:
        try:
            model = model.to('cpu')  # 先移到CPU
            quantized_model = quantize(model, weights=qint8, activations=qint8)
            if quantized_model is not None:
                model = quantized_model
            else:
                logging.warning("量化失败，使用原始模型")
        except Exception as e:
            logging.error(f"量化时发生错误: {e}")
            logging.warning("使用原始模型继续")
    
    model.eval()
    
    # 准备数据集
    dataset = TextDatasetFactory.get_dataset(
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
    
    # 加载图像字典(如果需要)
    img_dict = None
    print(config['img_dict_path'])
    if config['img_dict_path']:
        with open(config['img_dict_path'], 'rb') as f:
            img_dict = pickle.load(f)
    
    # 提取embeddings
    embeddings = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 3:  # (image, text, image_name)
                _, text, image_names = batch
                if img_dict:
                    target = torch.tensor([img_dict[name] for name in image_names]).to(args.device)
            else:  # (text, target)
                text, target = batch
            
            if config['text_prompt']:
                text = [config['text_prompt'].format(t) for t in text]
            
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(text, args.device)
            }
            
            model.to(args.device)
            current_embeddings = model(inputs)[ModalityType.TEXT]
            
            if embeddings:
                embeddings[ModalityType.TEXT] = torch.cat([embeddings[ModalityType.TEXT], current_embeddings], dim=0)
            else:
                embeddings[ModalityType.TEXT] = current_embeddings
            
            del current_embeddings
            logging.info(f"Processed batch: {batch_idx}")
    
    # 保存embeddings
    Path(args.embedding_dir).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {'text_embeddings': embeddings[ModalityType.TEXT]},
        args.embedding_dir
    )
    logging.info(f"Text embeddings saved to {args.embedding_dir}")

if __name__ == "__main__":
    main() 