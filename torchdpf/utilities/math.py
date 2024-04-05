import torch

def wrap_angle(angle): 
    return ((angle - torch.pi) % (2 * torch.pi)) - torch.pi