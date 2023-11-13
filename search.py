import logging
import torch
import data
import torchvision
import torchmetrics

from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA

from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from datasets.imagenet import ImageNetDataset
from metrics.search import Search

logging.basicConfig(level=logging.INFO, force=True)


lora = False
linear_probing = False
device = "cuda:0" if torch.cuda.is_available() else "cpu"
load_head_post_proc_finetuned = True
datadir = "/data/yx/ImageBind/.datasets/imagenet"
lora_dir = ''

assert not (linear_probing and lora), \
            "Linear probing is a subset of LoRA training procedure for ImageBind. " \
            "Cannot set both linear_probing=True and lora=True. "

if lora and not load_head_post_proc_finetuned:
    # Hack: adjust lora_factor to the `max batch size used during training / temperature` to compensate missing norm
    lora_factor = 12 / 0.07
else:
    # This assumes proper loading of all params but results in shift from original dist in case of LoRA
    lora_factor = 1


# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
if lora:
    model.modality_trunks.update(
        LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
                                        layer_idxs={ModalityType.TEXT: [ 1, 2, 3, 4, 5, 6, 7, 8],
                                                    ModalityType.VISION: [1, 2, 3, 4, 5, 6, 7, 8]},
                                        modality_names=[ModalityType.TEXT, ModalityType.VISION]))

    # Load LoRA params if found
    LoRA.load_lora_modality_trunks(model.modality_trunks,
                                   checkpoint_dir=lora_dir)

    if load_head_post_proc_finetuned:
        # Load postprocessors & heads
        load_module(model.modality_postprocessors, module_name="postprocessors",
                    checkpoint_dir=lora_dir)
        load_module(model.modality_heads, module_name="heads",
                    checkpoint_dir=lora_dir)
elif linear_probing:
    # Load heads
    load_module(model.modality_heads, module_name="heads",
                checkpoint_dir=lora_dir)

model.eval()
model.to(device)


def run_inference():
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

    
    test_ds = ImageNetDataset(datadir=datadir, split="val", device=device, transform=data_transform)
    test_dl = DataLoader(dataset=test_ds, batch_size=64, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True)
    
    test_search = Search(num_classes=1000, device=device, target_classes=[test_ds.text_list.index('stingray'),
                                         test_ds.text_list.index('cock'),
                                         test_ds.text_list.index('hen')])


    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_dl):
            if torch.min(target).item() > test_ds.text_list.index('hen') or torch.max(target).item() < test_ds.text_list.index('stingray'):
                continue

            # feats_a = [model({class_a[0]: data_a_i}) for data_a_i in data_a]
            # feats_a_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_a], dim=0)
            # # class_b could be any modality
            # feats_b = [model({class_b[idx]: data_b_i}) for idx, data_b_i in enumerate(data_b)]
            # feats_b_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_b], dim=0)

            # match_value = feats_a_tensor @ feats_b_tensor.T

            x = x.to(device)
            target = target.to(device)
            inputs = {
                ModalityType.VISION: x,
                ModalityType.TEXT: data.load_and_transform_text(test_ds.text_list, device),
            }
            
            embeddings = model(inputs)
            match_value_1 = embeddings[ModalityType.VISION]@embeddings[ModalityType.TEXT].T * (lora_factor if lora else 1)

            result_1 = torch.softmax(match_value_1, dim=-1)
            test_search.update(result_1,target)
        
        test_search.compute()
    
    print(test_search.pricision)
    print(test_search.recall)
        


if __name__ == "__main__":
    run_inference()

