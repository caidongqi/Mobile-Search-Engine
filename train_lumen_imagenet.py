# Based on PyTorch Lightning Tutorial 13 -
# SSL : https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html
# Modified by Fares Abawi (@fabawi).
import copy
import logging
import os
import argparse

try:
    import comet_ml
except ImportError:
    comet_ml = None
try:
    import wandb
except ImportError:
    wandb = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    logging.warning("Matplotlib not installed. This is not needed if you run this script as --headless")

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
from data import load_and_transform_text

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision import transforms
from datasets.imagenet import ImageNetDataset
import data
from models import lumen6_model
from models import lumen_model
from models import lora_lumen as LoRA
#from models.imagebind_model import ModalityType, load_module, save_module
from models.lumen6_model import ModalityType, load_module,save_module
from models.lumen_model import ModalityType, load_module,save_module
logging.basicConfig(level=logging.INFO, force=True)


# Logging settings
LOG_ON_STEP = True
LOG_ON_EPOCH = True


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


class ImageBindTrain(L.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=1e-4, max_epochs=500, batch_size=32, num_workers=4, seed=42, 
                 self_contrast=False, temperature=0.07,  momentum_betas=(0.9, 0.95), 
                 lora=False, lora_rank=4, lora_checkpoint_dir="./.checkpoints/lora",
                 lora_layer_idxs=None, lora_modality_names=None,
                 linear_probing=False, text_list=[]
                 ):
        super().__init__()
        assert not (linear_probing and lora), \
            "Linear probing is a subset of LoRA training procedure for ImageBind. " \
            "Cannot set both linear_probing=True and lora=True. " \
            "Linear probing stores params in lora_checkpoint_dir"
        self.save_hyperparameters()
        self.text_list = text_list
        # Load full pretrained ImageBind model
        self.model = lumen6_model.imagebind_huge(pretrained=True,vision_num_blocks_1=31,vision_num_blocks_2=1,text_num_blocks=24)
      

        if lora:
            torch.autograd.set_detect_anomaly(True)
            for modality_preprocessor in self.model.modality_preprocessors.children():
                modality_preprocessor.requires_grad_(False)
            for modality_trunk in self.model.modality_trunks.children():
                modality_trunk.requires_grad_(True)
            # for modality_trunk in self.model.modality_trunks_2.children():
            #     modality_trunk.requires_grad_(False)
            self.model.modality_trunks.update(LoRA.apply_lora_modality_trunks(self.model.modality_trunks, rank=lora_rank,
                                                                              layer_idxs=lora_layer_idxs,
                                                                              modality_names=lora_modality_names))
            LoRA.load_lora_modality_trunks(self.model.modality_trunks, checkpoint_dir=lora_checkpoint_dir, postfix = "_trunk_last")
            # self.model.modality_trunks_2.update(LoRA.apply_lora_modality_trunks(self.model.modality_trunks_2, rank=lora_rank,
            #                                                                   layer_idxs=lora_layer_idxs,
            #                                                                   modality_names=lora_modality_names))
            # LoRA.load_lora_modality_trunks(self.model.modality_trunks_2, checkpoint_dir=lora_checkpoint_dir, postfix = "_trunk2_last")
            # # Load postprocessors & heads
            # load_module(self.model.modality_postprocessors, module_name="preprocessors",
            #             checkpoint_dir=lora_checkpoint_dir)
            load_module(self.model.modality_postprocessors, module_name="postprocessors",
                        checkpoint_dir=lora_checkpoint_dir)
            load_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=lora_checkpoint_dir)
        elif linear_probing:
            for modality_preprocessor in self.model.modality_preprocessors.children():
                modality_preprocessor.requires_grad_(False)
            for modality_trunk in self.model.modality_trunks.children():
                modality_trunk.requires_grad_(False)
            for modality_postprocessor in self.model.modality_postprocessors.children():
                modality_postprocessor.requires_grad_(False)

            load_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=lora_checkpoint_dir)
            for modality_head in self.model.modality_heads.children():
                modality_head.requires_grad_(False)
                final_layer = list(modality_head.children())[-1]
                final_layer.requires_grad_(True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, 
                                betas=self.hparams.momentum_betas)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]
    
    
    
    def info_nce_loss(self, batch, mode="train"):
        # data_a, data_b = batch
        # #data_a=[data_a]
        # text = [self.text_list[i] for i in data_b.tolist()]
        # data_b = data.load_and_transform_text(text, self.device)
        # data_b = [data_b]
        # feats = self.model({ModalityType.VISION: data_a,ModalityType.TEXT: data_b}) 
        # feats_a_tensor=feats[ModalityType.VISION]
        # # class_b could be any modality
        # feats_b_tensor = feats[ModalityType.TEXT]
        data_a, data_b = batch
        data_a = [data_a]
        text = [self.text_list[i] for i in data_b.tolist()]
        data_b = data.load_and_transform_text(text, self.device)
        data_b = [data_b]

        # feats = [self.model({ModalityType.VISION: [data_a[i].unsqueeze(0)],ModalityType.TEXT: data_b })for i in range(data_a.shape[0]) ]
        # feats_a_tensor=torch.cat([dict_[ModalityType.VISION] for dict_ in feats],dim=0)
        # #feats_b_tensor=torch.cat([list(dict_[ModalityType.TEXT].values())[0] for dict_ in feats],dim=0)
        # feats_b_tensor=torch.cat([dict_[ModalityType.TEXT] for dict_ in feats],dim=0)
        
        feats_a = [self.model({ModalityType.VISION: data_a_i}) for data_a_i in data_a]
        feats_a_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_a], dim=0)
        # class_b could be any modality
        feats_b = [self.model({ModalityType.TEXT: data_b_i}) for data_b_i in data_b]
        feats_b_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_b], dim=0)

        
        if self.hparams.self_contrast:
            feats_a_b_tensor = torch.cat([feats_a_tensor.chunk(2)[0], feats_b_tensor], dim=0)
            feats_tensors = [feats_a_tensor, feats_a_b_tensor]
            temperatures = [1, self.hparams.temperature]
            contrast = ["self", "cross"]
        else:
            feats_a_b_tensor = torch.cat([feats_a_tensor, feats_b_tensor], dim=0)
            feats_tensors = [feats_a_b_tensor]
            temperatures = [self.hparams.temperature]
            contrast = ["cross"]

        # Accumulate self-contrastive loss for image and its augmentation, and modailty with image
        dual_nll = False
        for feats_idx, feats_tensor in enumerate(feats_tensors):
            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(feats_tensor[:, None, :], feats_tensor[None, :, :], dim=-1)
            # Mask out cosine similarity to itself
            self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
            cos_sim.masked_fill_(self_mask, -9e15)
            # Find positive example -> batch_size//2 away from the original example
            pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
            # InfoNCE loss
            cos_sim = cos_sim / temperatures[feats_idx]
            nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
            nll = nll.mean()
            if not dual_nll:
                dual_nll = nll
            else:
                dual_nll += nll
                dual_nll /= 2
            # Logging loss
            self.log(mode + "_loss_" + contrast[feats_idx], nll, prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)
            # Get ranking position of positive example
            comb_sim = torch.cat(
                [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
                dim=-1,
            )
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
            # Logging ranking metrics
            self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean(), prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)
            self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean(), prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)
            self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean(), prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)

        self.log(mode + "_loss", dual_nll, prog_bar=True,
                 on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)
        return dual_nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
    #     data_a, target = batch
    #   #  data_a = [data_a]
    #     text = copy.deepcopy(test_dataset.text_list)
    #     data_b = data.load_and_transform_text(text, self.device)
    #     data_b = [data_b]
    #     #data_a=[data_a]
        data_a, target = batch
        data_a = [data_a]
        text = copy.deepcopy(test_dataset.text_list)
        data_b = data.load_and_transform_text(text, self.device)
        data_b = [data_b]
        
        
        # feats = [self.model({ModalityType.VISION: [data_a[i].unsqueeze(0)],ModalityType.TEXT: data_b })for i in range(data_a.shape[0]) ]
        # feats_a_tensor=torch.cat([dict_[ModalityType.VISION] for dict_ in feats],dim=0)
        # #feats_b_tensor=torch.cat([list(dict_[ModalityType.TEXT].values())[0] for dict_ in feats],dim=0)
        # feats_b_tensor=torch.cat([dict_[ModalityType.TEXT] for dict_ in feats],dim=0)
        
        # class_a is always "vision" according to ImageBind
        feats_a = [self.model({ModalityType.VISION: data_a_i}) for data_a_i in data_a]
        feats_a_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_a], dim=0)
        # class_b could be any modality
        feats_b = [self.model({ModalityType.TEXT: data_b_i}) for data_b_i in data_b]
        feats_b_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_b], dim=0)

        
        # class_b could be any modality
        #feats_a_tensor = feats[ModalityType.VISION]
        #feats_b_tensor = feats[ModalityType.TEXT]
        match_value = feats_a_tensor @ feats_b_tensor.T

        result = torch.softmax(match_value, dim=-1)
        _, predicted = torch.max(result, -1)
        correct = predicted.eq(target).sum()
        test_correct = correct.item()
        test_total = target.size(0)
        self.log("val" + "_acc_top1", test_correct / test_total, prog_bar=True,
                on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)
        

    def on_validation_epoch_end(self):
        if self.hparams.lora:
            # Save LoRA checkpoint
            LoRA.save_lora_modality_trunks(self.model.modality_trunks,checkpoint_dir=self.hparams.lora_checkpoint_dir, postfix= "_trunk1_last")
            #LoRA.save_lora_modality_trunks(self.model.modality_trunks_2,checkpoint_dir=self.hparams.lora_checkpoint_dir, postfix= "_trunk2_last")
            #LoRA.save_lora_modality_trunks(self.model.modality_trunks_2, checkpoint_dir=self.hparams.lora_checkpoint_dir)
            # Save postprocessors & heads
            # save_module(self.model.modality_preprocessors, module_name="preprocessors",
            #             checkpoint_dir=self.hparams.lora_checkpoint_dir)
            save_module(self.model.modality_postprocessors, module_name="postprocessors",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)
            save_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)
        elif self.hparams.linear_probing:
            # Save postprocessors & heads
            save_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)

    
def parse_args():
    parser = argparse.ArgumentParser(description="Train the ImageBind model with PyTorch Lightning and LoRA.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to use for training ('cpu' or 'cuda')")
    parser.add_argument("--datasets_dir", type=str, default="/data/air/pc/Mobile-Search-Engine/.datasets/one_imagenet",
                        help="Directory containing the datasets")
    parser.add_argument("--full_model_checkpoint_dir", type=str, default="./.checkpoints/full/lumen",
                        help="Directory to save the full model checkpoints")
    parser.add_argument("--full_model_checkpointing",action="store_true", help="Save full model checkpoints")
    parser.add_argument("--loggers", type=str, nargs="+", choices=["tensorboard", "wandb", "comet", "mlflow"],
                        help="Loggers to use for logging")
    parser.add_argument("--loggers_dir", type=str, default="./.logs/lumen", help="Directory to save the logs")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (Don't plot samples on start)")

    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--momentum_betas", nargs=2, type=float, default=[0.9, 0.95],
                        help="Momentum beta 1 and 2 for Adam optimizer")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature parameter for InfoNCE loss")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--self_contrast", action="store_true", help="Use self-contrast on the image modality")

    parser.add_argument("--lora", action="store_true", default=True,help="Use LoRA")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of LoRA layers")
    parser.add_argument("--lora_checkpoint_dir", type=str, default="./.checkpoints/lora31-1",
                        help="Directory to save LoRA checkpoint")
    parser.add_argument("--lora_modality_names", nargs="+", type=str, default=["vision", "text"],
                        choices=["vision", "text", "audio", "thermal", "depth", "imu"],
                        help="Modality names to apply LoRA")
    parser.add_argument("--lora_layer_idxs", nargs="+", type=int,
                        help="Layer indices to apply LoRA")
    parser.add_argument("--lora_layer_idxs_vision", nargs="+", type=int,
                        help="Layer indices to apply LoRA for vision modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_text", nargs="+", type=int,
                        help="Layer indices to apply LoRA for text modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_audio", nargs="+", type=int,
                        help="Layer indices to apply LoRA for audio modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_thermal", nargs="+", type=int,
                        help="Layer indices to apply LoRA for thermal modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_depth", nargs="+", type=int,
                        help="Layer indices to apply LoRA for depth modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_imu", nargs="+", type=int,
                        help="Layer indices to apply LoRA for imu modality. Overrides lora_layer_idxs if specified")

    parser.add_argument("--linear_probing", action="store_true",
                        help="Freeze model and train the last layers of the head for each modality.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create loggers
    loggers = []
    for logger in args.loggers if args.loggers is not None else []:
        if logger == "wandb":
            wandb.init(project="imagebind", config=args)
            wandb_logger = pl_loggers.WandbLogger(
                save_dir=args.loggers_dir,
                name="imagebind")
            loggers.append(wandb_logger)
        elif logger == "tensorboard":
            tensorboard_logger = pl_loggers.TensorBoardLogger(
                save_dir=args.loggers_dir,
                name="imagebind")
            loggers.append(tensorboard_logger)
        elif logger == "comet":
            comet_logger = pl_loggers.CometLogger(
                save_dir=args.loggers_dir,
                api_key=os.environ["COMET_API_KEY"],
                workspace=os.environ["COMET_WORKSPACE"],
                project_name=os.environ["COMET_PROJECT_NAME"],
                experiment_name=os.environ.get("COMET_EXPERIMENT_NAME", None),
            )
            loggers.append(comet_logger)
        elif logger == "mlflow":
            mlflow_logger = pl_loggers.MLFlowLogger(
                save_dir=args.loggers_dir,
                experiment_name=os.environ["MLFLOW_EXPERIMENT_NAME"],
                tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
                run_name="imagebind"
            )
            loggers.append(mlflow_logger)
        else:
            raise ValueError(f"Unknown logger: {logger}")

    # Set experiment properties
    seed_everything(args.seed, workers=True)
    torch.backends.cudnn.determinstic = True
    device_name = args.device  # "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    data_transform = transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    train_dataset = ImageNetDataset(transform=data_transform,split='train',device=device,datadir=args.datasets_dir)
    test_dataset = ImageNetDataset(transform=data_transform,split='val',device=device,datadir=args.datasets_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        num_workers=args.num_workers,
    )

    # Parse indices of layers to apply LoRA
    lora_layer_idxs = {}
    lora_modality_names = []
    modalities = ["vision", "text", "audio", "thermal", "depth", "imu"]
    for modality_name in args.lora_modality_names:
        if modality_name in modalities:
            modality_type = getattr(ModalityType, modality_name.upper())
            lora_layer_idxs[modality_type] = getattr(args, f'lora_layer_idxs_{modality_name}', None)
            if not lora_layer_idxs[modality_type]:
                lora_layer_idxs[modality_type] = None
            lora_modality_names.append(modality_type)
        else:
            raise ValueError(f"Unknown modality name: {modality_name}")

    # Train dataset
    model = ImageBindTrain(max_epochs=args.max_epochs, batch_size=args.batch_size, lr=args.lr,
                           weight_decay=args.weight_decay, momentum_betas=args.momentum_betas,
                           temperature=args.temperature,
                           num_workers=args.num_workers, self_contrast=args.self_contrast,
                           lora=args.lora, lora_rank=args.lora_rank, lora_checkpoint_dir=args.lora_checkpoint_dir,
                           lora_layer_idxs=lora_layer_idxs if lora_layer_idxs else None,
                           lora_modality_names=lora_modality_names if lora_modality_names else None,
                           linear_probing=args.linear_probing, text_list=test_dataset.text_list)

    if args.full_model_checkpointing:
        checkpointing = {"enable_checkpointing": args.full_model_checkpointing,
                         "callbacks": [ModelCheckpoint(monitor="val_loss", dirpath=args.full_model_checkpoint_dir,
                                                        filename="imagebind-{epoch:02d}-{val_loss:.2f}",
                                                        save_last=True, mode="min")]}
    else:
        checkpointing = {"enable_checkpointing": args.full_model_checkpointing,}
    #devices=1 if ":" not in device_name else [int(device_name.split(":")[1])]
    
    trainer = Trainer(accelerator="gpu" if "cuda" in device_name else "cpu",
                      devices=[5], deterministic=True,
                      max_epochs=args.max_epochs, gradient_clip_val=args.gradient_clip_val,
                      logger=loggers if loggers else None, **checkpointing, strategy='ddp_find_unused_parameters_true')

    trainer.fit(model, train_loader, val_loader)


