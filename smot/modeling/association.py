import torch
from torch import nn

class AssociationModel(nn.Module):
    def __init__(self):
        pass

    def associate(new_tracks: torch.Tensor):
        '''
        - Arguments:
            - tracks: np.array of shape (nb_tracks, 5)
        
        - Returns:
            - predictions: np.array of shape (nb_tracks,)
        '''
        pass

    def update_model(tracks: torch.Tensor, preds: torch.Tensor):
        '''
        - Arguments:
            - tracks: torch.Tensor of shape (nb_tracks, track_max_size, 5)
            - preds: torch.Tensor of shape (nb_tracks,)
        '''
        pass