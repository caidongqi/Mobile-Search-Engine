import torch

def count_parameters(model):
    """
    Counts the total and trainable parameters of a torch.nn.Module.

    Parameters:
    - model: An instance of torch.nn.Module.

    Returns:
    - A tuple containing the total number of parameters and the number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params

# Example usage:
# Define your model (e.g., model = MyModel())
# total_params, trainable_params = count_parameters(model)
# print(f"Total Parameters: {total_params}")
# print(f"Trainable Parameters: {trainable_params}")
