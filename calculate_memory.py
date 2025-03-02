import torch
from models import imagebind_model
from models.imagebind_model import ModalityType, load_module

def calculate_modality_memory_usage(model):
    total_memory_mb = 0

    for modality_key in ModalityType.__dict__.keys():
        if not modality_key.startswith('__'):  # Skip special attributes
            modality_key = modality_key.lower()
            preprocessors = getattr(model.modality_preprocessors, modality_key, None)
            trunks = getattr(model.modality_trunks, modality_key, None)
            heads = getattr(model.modality_heads, modality_key, None)
            postprocessors = getattr(model.modality_postprocessors, modality_key, None)

            num_params = 0

            if preprocessors:
                num_params += sum(p.numel() for p in preprocessors.parameters() if p.requires_grad)

            if trunks:
                num_params += sum(p.numel() for p in trunks.parameters() if p.requires_grad)

            if heads:
                num_params += sum(p.numel() for p in heads.parameters() if p.requires_grad)

            if postprocessors:
                num_params += sum(p.numel() for p in postprocessors.parameters() if p.requires_grad)

            memory_bytes = num_params * 4  # Assuming float32, each parameter takes 4 bytes
            memory_mb = memory_bytes / (1024 ** 2)
            total_memory_mb += memory_mb

            print(f"{modality_key.upper()} total memory usage: {memory_mb:.2f} MB.")

    print(f"Total model memory usage for all modalities: {total_memory_mb:.2f} MB")

# Example usage:
model = imagebind_model.imagebind_huge(pretrained=True)
calculate_modality_memory_usage(model)