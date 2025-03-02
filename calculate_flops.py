import torch
import torch.nn as nn
from torchprofile import profile_macs
from models import imagebind_model
from models.imagebind_model import ModalityType, load_module

# 模型实例化
model = imagebind_model.imagebind_huge(pretrained=True)


# 计算单个模态的FLOPs和参数数量的函数
def calculate_macs_and_params(modality_type, input_tensor):
    # 将单个模态的输入包装成字典
    inputs = {modality_type: input_tensor}
    
    # 计算FLOPs
    macs = profile_macs(model, inputs)
    
    # 计算参数数量
    # 只计算该模态相关部分的参数数量
    params = sum(p.numel() for n, p in model.named_parameters() if modality_type in n)
    
    return macs, params

# 创建每个模态的示例输入
batch_size = 1  # 批量大小
vision_input = torch.randn(batch_size, 3, 224, 224)  # 示例视觉输入
text_input = torch.randint(0, 49408, (batch_size, 77))  # 示例文本输入
audio_input = torch.randn(batch_size, 1, 128, 204)  # 示例音频输入
depth_input = torch.randn(batch_size, 1, 224, 224)  # 示例深度输入
thermal_input = torch.randn(batch_size, 1, 224, 224)  # 示例热成像输入
imu_input = torch.randn(batch_size, 6, 2000)  # 示例IMU输入

# 计算每个模态的FLOPs和参数数量
vision_macs, vision_params = calculate_macs_and_params(ModalityType.VISION, vision_input)
text_macs, text_params = calculate_macs_and_params(ModalityType.TEXT, text_input)
audio_macs, audio_params = calculate_macs_and_params(ModalityType.AUDIO, audio_input)
depth_macs, depth_params = calculate_macs_and_params(ModalityType.DEPTH, depth_input)
thermal_macs, thermal_params = calculate_macs_and_params(ModalityType.THERMAL, thermal_input)
imu_macs, imu_params = calculate_macs_and_params(ModalityType.IMU, imu_input)

print(f"Vision FLOPs: {vision_macs / 1e9} GFLOPs, Parameters: {vision_params / 1e6} Million")
print(f"Text FLOPs: {text_macs / 1e9} GFLOPs, Parameters: {text_params / 1e6} Million")
print(f"Audio FLOPs: {audio_macs / 1e9} GFLOPs, Parameters: {audio_params / 1e6} Million")
print(f"Depth FLOPs: {depth_macs / 1e9} GFLOPs, Parameters: {depth_params / 1e6} Million")
print(f"Thermal FLOPs: {thermal_macs / 1e9} GFLOPs, Parameters: {thermal_params / 1e6} Million")
print(f"IMU FLOPs: {imu_macs / 1e9} GFLOPs, Parameters: {imu_params / 1e6} Million")

# 将输入包装成字典
inputs = {
    ModalityType.VISION: vision_input,
    ModalityType.TEXT: text_input,
    ModalityType.AUDIO: audio_input,
    ModalityType.DEPTH: depth_input,
    ModalityType.THERMAL: thermal_input,
    ModalityType.IMU: imu_input,
}
macs = profile_macs(model, inputs)

# 计算参数数量
params = sum(p.numel() for p in model.parameters())

print(f"FLOPs: {macs / 1e9} GFLOPs")
print(f"Parameters: {params / 1e6} Million")