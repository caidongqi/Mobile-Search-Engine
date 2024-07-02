import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import ImageNet
from torchvision.datasets import CocoDetection
from models.imagebind_model import ModalityType
import data
import torch

class CoCoDataset(CocoDetection):
    def __init__(self,transform,device='cpu',annFile: str="",datadir=""):
        self.text_list =['person',
'bicycle',
'car',
'motorcycle',
'airplane',
'bus',
'train',
'truck',
'boat',
'traffic light',
'fire hydrant',
'NULL000000',
'stop sign',
'parking meter',
'bench',
'bird',
'cat',
'dog',
'horse',
'sheep',
'cow',
'elephant',
'bear',
'zebra',
'giraffe',
'NULL000000',
'backpack',
'umbrella',
'NULL000000',
'NULL000000',
'handbag',
'tie',
'suitcase',
'frisbee',
'skis',
'snowboard',
'sports ball',
'kite',
'baseball bat',
'baseball glove',
'skateboard',
'surfboard',
'tennis racket',
'bottle',
'NULL000000',
'wine glass',
'cup',
'fork',
'knife',
'spoon',
'bowl',
'banana',
'apple',
'sandwich',
'orange',
'broccoli',
'carrot',
'hot dog',
'pizza',
'donut',
'cake',
'chair',
'couch',
'potted plant',
'bed',
'NULL000000',
'dining table',
'NULL000000',
'NULL000000',
'toilet',
'NULL000000',
'tv',
'laptop',
'mouse',
'remote',
'keyboard',
'cell phone',
'microwave',
'oven',
'toaster',
'sink',
'refrigerator',
'NULL000000',
'book',
'clock',
'vase',
'scissors',
'teddy bear',
'hair drier',
'toothbrush',] 
        self.device = device
        super().__init__(datadir, annFile=annFile, transform=transform)
        
    def __getitem__(self, index: int):
        images, target = super().__getitem__(index)
        # category=[]
        # for item in target:
        #     category.append(item['category_id'])
        # # print(f"Image Shape: {images.shape}")
        # # print(f"Target Shape: {category}")
        # category = list(set(category))
        # target_length=20
        # arr=category
        # if len(arr) < target_length:
        #    arr += [0] * (target_length - len(arr))
        # # 如果数组长度超过目标长度，截断多余的元素
        # elif len(arr) > target_length:
            # arr = arr[:target_length]
        return images