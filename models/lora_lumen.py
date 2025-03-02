# Sheng Wang at Feb 22 2023
# Based on LoRA-ViT: https://github.com/JamesQFreeman/LoRA-ViT/blob/main/lora.py
# Modified by Fares Abawi (@fabawi).

import logging
import os
import math
from typing import Optional, List, Dict, Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from torch.nn.parameter import Parameter

torch.autograd.set_detect_anomaly(True)

# 其他代码

from models.transformer import SimpleTransformer
from models.transformer2 import SimpleTransformer2

def apply_lora_modality_trunks(modality_trunks: Dict[str, SimpleTransformer], rank: int,
                               layer_idxs: Optional[Dict[SimpleNamespace, List[int]]] = None,
                               modality_names: List[SimpleNamespace] = None):
    if modality_names is None:
        modality_names = list(modality_trunks.keys())
    if layer_idxs is None:
        layer_idxs = {}
    # return nn.ModuleDict({modality_name: LoRA_SimpleTransformer(modality_trunk, rank, layer_idxs.get(modality_name, None)) for
    #                       modality_name, modality_trunk in modality_trunks.items() if modality_name in modality_names})
    return nn.ModuleDict({
    modality_name: (LoRA_SimpleTransformer2 if modality_name == 'vision' else LoRA_SimpleTransformer)(
        modality_trunk, rank, layer_idxs.get(modality_name, None)
    ) 
    for modality_name, modality_trunk in modality_trunks.items() if modality_name in modality_names
})

# def save_lora_modality_trunks(modality_trunks: Dict[str, SimpleTransformer],
#                               checkpoint_dir: str = "./.checkpoints/lora", postfix: str = "_last", extension: str = "safetensors"):
#     for modality_name, modality_trunk in modality_trunks.items():
#         try:
#             if isinstance(modality_trunk, LoRA_SimpleTransformer):
#                 modality_trunk.save_lora_parameters(os.path.join(checkpoint_dir, f"imagebind-lora-{modality_name}{postfix}.{extension}"))
#                 logging.info(f"Saved LoRA parameters for modality {modality_name} to {checkpoint_dir}.")
#         except FileNotFoundError:
#             logging.warning(f"Could not save LoRA parameters for modality {modality_name} to {checkpoint_dir}.")

# def save_lora_modality_trunks(modality_trunks_1: Dict[str, SimpleTransformer],modality_trunks_2: Dict[str, SimpleTransformer],
#                               checkpoint_dir: str = "./.checkpoints/lora", postfix: str = "_last", extension: str = "safetensors"):
#     dict={}
#     for (modality_name_1, modality_trunk_1), (modality_name_2, modality_trunk_2) in zip(modality_trunks_1.items(), modality_trunks_2.items()):  
#         try:  
#             if isinstance(modality_trunk_1, LoRA_SimpleTransformer):  
#                 dict = modality_trunk_1.save_lora_parameters()  
#                 logging.info(f"Saved LoRA parameters for modality {modality_name_1} to {checkpoint_dir}.")  
                
#             if isinstance(modality_trunk_2, LoRA_SimpleTransformer):  
#                 modality_trunk_2.save_lora_parameters2(os.path.join(checkpoint_dir, f"imagebind-lora-{modality_name_2}{postfix}.{extension}"), num1=len(modality_trunk_1.w_As), merged_dict=dict)  
#                 logging.info(f"Saved LoRA parameters for modality {modality_name_2} to {checkpoint_dir}.")  
                
#         except FileNotFoundError:  
#             logging.warning(f"Could not save LoRA parameters for modalities {modality_name_1} and/or {modality_name_2} to {checkpoint_dir}.")  

def save_lora_modality_trunks(modality_trunks: Dict[str, SimpleTransformer],
                              checkpoint_dir: str = "./.checkpoints/lora", postfix: str = "_last", extension: str = "safetensors"):
    for modality_name, modality_trunk in modality_trunks.items():
        try:
            if isinstance(modality_trunk, (LoRA_SimpleTransformer2 if modality_name == 'vision' else LoRA_SimpleTransformer)):
                if modality_name=='vision':
                    postfix="_block1_last"
                    modality_trunk.save_lora_parameters(modality_trunk.w_As1,modality_trunk.w_Bs1,os.path.join(checkpoint_dir, f"imagebind-lora-{modality_name}{postfix}.{extension}"))
                    postfix="_block2_last"
                    modality_trunk.save_lora_parameters(modality_trunk.w_As2,modality_trunk.w_Bs2,os.path.join(checkpoint_dir, f"imagebind-lora-{modality_name}{postfix}.{extension}"))
                
                else:
                    postfix="_last"
                    modality_trunk.save_lora_parameters(os.path.join(checkpoint_dir, f"imagebind-lora-{modality_name}{postfix}.{extension}"))
                    logging.info(f"Saved LoRA parameters for modality {modality_name} to {checkpoint_dir}.")
                    
        except FileNotFoundError:
            logging.warning(f"Could not save LoRA parameters for modality {modality_name} to {checkpoint_dir}.")


def load_lora_modality_trunks(modality_trunks: Dict[str, SimpleTransformer],
                              checkpoint_dir: str = "./.checkpoints/lora", postfix: str = "_last", extension: str = "safetensors"):

    for modality_name, modality_trunk in modality_trunks.items():
        try:
            if isinstance(modality_trunk, (LoRA_SimpleTransformer2 if modality_name == 'vision' else LoRA_SimpleTransformer)):
                if modality_name=='vision':
                    postfix="_block1_last"
                    modality_trunk.load_lora_parameters(modality_trunk.w_As1,modality_trunk.w_Bs1,os.path.join(checkpoint_dir, f"imagebind-lora-{modality_name}{postfix}.{extension}"))
                    logging.info(f"Loaded LoRA parameters for modality {modality_name} {postfix}from {checkpoint_dir}.")
       
                    postfix="_block2_last"
                    modality_trunk.load_lora_parameters(modality_trunk.w_As2,modality_trunk.w_Bs2,os.path.join(checkpoint_dir, f"imagebind-lora-{modality_name}{postfix}.{extension}"))
                    logging.info(f"Loaded LoRA parameters for modality {modality_name}{postfix} from {checkpoint_dir}.")
       
                else:
                   postfix="_last"
                   modality_trunk.load_lora_parameters(os.path.join(checkpoint_dir, f"imagebind-lora-{modality_name}{postfix}.{extension}"))
                   logging.info(f"Loaded LoRA parameters for modality {modality_name} from {checkpoint_dir}.")
        except FileNotFoundError:
            logging.warning(f"Could not find LoRA parameters for modality {modality_name} in {checkpoint_dir}.")
            logging.warning("If you are training the sub-model from scratch, this is expected.")
            logging.warning("If you are loading parts of a pre-trained model, this is expected for some modalities.")


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, **kwargs):

        x = self.w(x, attn_mask=attn_mask) + self.w_b(self.w_a(x))
        return x


class LoRA_SimpleTransformer_vision(nn.Module):
    """Applies low-rank adaptation to simple transformer with pytorch multihead attention.

    Args:
        transformer_model: a vision transformer model, see base_vit.py
        rank: rank of LoRA
        lora_layer_idxs: which layer we apply LoRA.

    Examples::
        >>> model = SimpleTransformer()
        >>> lora_model = LoRA_SimpleTransformer(model, rank=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, transformer_model: SimpleTransformer, rank: int, lora_layer_idxs: Optional[List[int]] = None):
        super(LoRA_SimpleTransformer_vision, self).__init__()

        assert rank > 0
        self.base_dim = transformer_model.blocks1[0].attn.in_proj_bias.size()[0]//3
        dim = self.base_dim
        if lora_layer_idxs is not None:
            self.lora_layer_idxs1 = lora_layer_idxs
        else:
            self.lora_layer_idxs1 = list(range(len(transformer_model.blocks1)))
            
        if lora_layer_idxs is not None:
            self.lora_layer_idxs2 = lora_layer_idxs
        else:
            self.lora_layer_idxs2 = list(range(len(transformer_model.blocks2)))
        # create for storage, then we can init them or load weights
        self.w_As1 = []  # These are linear layers
        self.w_Bs1 = []
        self.w_As2 = []  # These are linear layers
        self.w_Bs2 = []
        # lets freeze first
        for param in transformer_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_idx, blk in enumerate(transformer_model.blocks1):
            # If we only want few lora layer instead of all
            if t_layer_idx not in self.lora_layer_idxs1:
                continue
            w_a_linear_qkv = nn.Linear(dim, rank, bias=False)
            w_b_linear_qkv = nn.Linear(rank, dim, bias=False)
            self.w_As1.append(w_a_linear_qkv)
            self.w_Bs1.append(w_b_linear_qkv)
            blk.prev_attn = blk.attn
            blk.attn = _LoRALayer(blk.prev_attn, w_a_linear_qkv, w_b_linear_qkv)

        for t_layer_idx, blk in enumerate(transformer_model.blocks2):
            # If we only want few lora layer instead of all
            if t_layer_idx not in self.lora_layer_idxs2:
                continue
            w_a_linear_qkv = nn.Linear(dim, rank, bias=False)
            w_b_linear_qkv = nn.Linear(rank, dim, bias=False)
            self.w_As2.append(w_a_linear_qkv)
            self.w_Bs2.append(w_b_linear_qkv)
            blk.prev_attn = blk.attn
            blk.attn = _LoRALayer(blk.prev_attn, w_a_linear_qkv, w_b_linear_qkv)

        if self.training:
            self.reset_parameters()
        self.lora_model = transformer_model

    def save_lora_parameters(self,filename:str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensors if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As1)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As1[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs1[i].weight for i in range(num_layer)}
        
        merged_dict = {**a_tensors, **b_tensors}
        num_layer = len(self.w_As2)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As2[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs2[i].weight for i in range(num_layer)}
        merged_dict.update(a_tensors)
        merged_dict.update(b_tensors)
        os.makedirs(os.path.dirname(filename),exist_ok=True)
        save_file(merged_dict, filename)
        return merged_dict
    
    def save_lora_parameters2(self,filename: str,num1=31,merged_dict=[]) -> None:
        r"""Only safetensors is supported now.

        pip install safetensors if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i+num1:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i+num1:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        merged_dict.update(a_tensors)
        merged_dict.update(b_tensors)
        os.makedirs(os.path.dirname(filename),exist_ok=True)
        save_file(merged_dict, filename)
        
    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensors if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As1):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs1):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
            
            for i, w_A_linear in enumerate(self.w_As2):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs2):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)

    def reset_parameters(self) -> None:
        for w_A in self.w_As1:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs1:
            nn.init.zeros_(w_B.weight)
        for w_A in self.w_As2:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs2:
            nn.init.zeros_(w_B.weight)
            
    def forward(self, tokens: torch.Tensor, **kwargs) -> Tensor:
        return self.lora_model(tokens)

class LoRA_SimpleTransformer2(nn.Module):
    """Applies low-rank adaptation to simple transformer with pytorch multihead attention.

    Args:
        transformer_model: a vision transformer model, see base_vit.py
        rank: rank of LoRA
        lora_layer_idxs: which layer we apply LoRA.

    Examples::
        >>> model = SimpleTransformer()
        >>> lora_model = LoRA_SimpleTransformer(model, rank=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, transformer_model: SimpleTransformer2, rank: int, lora_layer_idxs: Optional[List[int]] = None):
        super(LoRA_SimpleTransformer2, self).__init__()

        assert rank > 0
        self.base_dim = transformer_model.blocks[0].attn.in_proj_bias.size()[0]//3
        dim = self.base_dim
        if lora_layer_idxs is not None:
            self.lora_layer_idxs1 = lora_layer_idxs
        else:
            self.lora_layer_idxs1 = list(range(len(transformer_model.blocks_1)))
        # create for storage, then we can init them or load weights
        self.w_As1 = []  # These are linear layers
        self.w_Bs1 = []

        # lets freeze first
        for param in transformer_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_idx, blk in enumerate(transformer_model.blocks_1):
            # If we only want few lora layer instead of all
            if t_layer_idx not in self.lora_layer_idxs1:
                continue
            w_a_linear_qkv = nn.Linear(dim, rank, bias=False)
            w_b_linear_qkv = nn.Linear(rank, dim, bias=False)
            self.w_As1.append(w_a_linear_qkv)
            self.w_Bs1.append(w_b_linear_qkv)
            blk.prev_attn = blk.attn
            blk.attn = _LoRALayer(blk.prev_attn, w_a_linear_qkv, w_b_linear_qkv)
        
        if lora_layer_idxs is not None:
            self.lora_layer_idxs2 = lora_layer_idxs
        else:
            self.lora_layer_idxs2 = list(range(len(transformer_model.blocks_2)))
        # create for storage, then we can init them or load weights
        self.w_As2 = []  # These are linear layers
        self.w_Bs2 = []

        # Here, we do the surgery
        for t_layer_idx, blk in enumerate(transformer_model.blocks_2):
            # If we only want few lora layer instead of all
            if t_layer_idx not in self.lora_layer_idxs2:
                continue
            w_a_linear_qkv = nn.Linear(dim, rank, bias=False)
            w_b_linear_qkv = nn.Linear(rank, dim, bias=False)
            self.w_As2.append(w_a_linear_qkv)
            self.w_Bs2.append(w_b_linear_qkv)
            blk.prev_attn = blk.attn
            blk.attn = _LoRALayer(blk.prev_attn, w_a_linear_qkv, w_b_linear_qkv)

        
        if self.training:
            self.reset_parameters()
        self.lora_model = transformer_model

    def save_lora_parameters(self,w_As,w_Bs,filename:str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensors if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": w_Bs[i].weight for i in range(num_layer)}
        
        merged_dict = {**a_tensors, **b_tensors}
        
        os.makedirs(os.path.dirname(filename),exist_ok=True)
        save_file(merged_dict, filename)
        return merged_dict
    
    # def save_lora_parameters2(self,filename: str,num1=31,merged_dict=[]) -> None:
    #     r"""Only safetensors is supported now.

    #     pip install safetensors if you do not have one installed yet.
    #     """

    #     assert filename.endswith(".safetensors")

    #     num_layer = len(self.w_As)  # actually, it is half
    #     a_tensors = {f"w_a_{i+num1:03d}": self.w_As[i].weight for i in range(num_layer)}
    #     b_tensors = {f"w_b_{i+num1:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
    #     merged_dict.update(a_tensors)
    #     merged_dict.update(b_tensors)
    #     os.makedirs(os.path.dirname(filename),exist_ok=True)
    #     save_file(merged_dict, filename)
        
    def load_lora_parameters(self, w_As,w_Bs,filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensors if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)

    def reset_parameters(self) -> None:
        for w_A in self.w_As1:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs1:
            nn.init.zeros_(w_B.weight)
        for w_A in self.w_As2:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs2:
            nn.init.zeros_(w_B.weight)

    def forward(self, tokens: torch.Tensor, **kwargs) -> Tensor:
        return self.lora_model(tokens)



class LoRA_SimpleTransformer(nn.Module):
    """Applies low-rank adaptation to simple transformer with pytorch multihead attention.

    Args:
        transformer_model: a vision transformer model, see base_vit.py
        rank: rank of LoRA
        lora_layer_idxs: which layer we apply LoRA.

    Examples::
        >>> model = SimpleTransformer()
        >>> lora_model = LoRA_SimpleTransformer(model, rank=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, transformer_model: SimpleTransformer, rank: int, lora_layer_idxs: Optional[List[int]] = None):
        super(LoRA_SimpleTransformer, self).__init__()

        assert rank > 0
        self.base_dim = transformer_model.blocks[0].attn.in_proj_bias.size()[0]//3
        dim = self.base_dim
        if lora_layer_idxs is not None:
            self.lora_layer_idxs = lora_layer_idxs
        else:
            self.lora_layer_idxs = list(range(len(transformer_model.blocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in transformer_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_idx, blk in enumerate(transformer_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_idx not in self.lora_layer_idxs:
                continue
            w_a_linear_qkv = nn.Linear(dim, rank, bias=False)
            w_b_linear_qkv = nn.Linear(rank, dim, bias=False)
            self.w_As.append(w_a_linear_qkv)
            self.w_Bs.append(w_b_linear_qkv)
            blk.prev_attn = blk.attn
            blk.attn = _LoRALayer(blk.prev_attn, w_a_linear_qkv, w_b_linear_qkv)

        if self.training:
            self.reset_parameters()
        self.lora_model = transformer_model

    def save_lora_parameters(self,filename:str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensors if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        merged_dict = {**a_tensors, **b_tensors}
        os.makedirs(os.path.dirname(filename),exist_ok=True)
        save_file(merged_dict, filename)
        return merged_dict
    
    def save_lora_parameters2(self,filename: str,num1=31,merged_dict=[]) -> None:
        r"""Only safetensors is supported now.

        pip install safetensors if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i+num1:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i+num1:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        merged_dict.update(a_tensors)
        merged_dict.update(b_tensors)
        os.makedirs(os.path.dirname(filename),exist_ok=True)
        save_file(merged_dict, filename)
        
    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensors if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, tokens: torch.Tensor, **kwargs) -> Tensor:
        return self.lora_model(tokens)
