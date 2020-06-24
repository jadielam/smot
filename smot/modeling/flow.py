import torch
from torch import nn

class FlowModule(nn.Module):
    def __init__(self):
        pass

    def forward(tracks: torch.Tensor):
        '''
        - Arguments:
            - tracks: torch.Tensor of shape (nb_tracks, tracks_max_size, 4)
        
        - Returns:
            - preds: torch.Tensor of shape (nb_tracks, 4)
        '''
        # TODO
        pass


def loss_fn(preds: torch.Tensor, gt: torch.Tensor):
    '''
    - Arguments:
        - preds: torch.Tensor of shape (nb_tracks, 4)
        - gt: torch.Tensor of shape (nb_tracks, 4)
    
    - Returns:
        - loss
    '''
    # TODO
    pass

class FlowModel(object):
    def __init__(self, lr = 0.001):
        self._flow_module = FlowModule()
        self._optimizer = torch.optim.Adam(self._flow_module.parameters(), lr = lr)
        self._loss_balancer = 0
        
    def _linear_prediction(self, tracks: torch.Tensor):
        '''
        - Arguments:
            - tracks: torch.Tensor of shape (nb_tracks, track_max_size, 4)
        
        - Returns:
            - preds: (nb_tracks, 4)
        '''
        # TODO
        pass

    def predict(self, tracks: torch.Tensor):
        '''
        Predicts the next location of the given tracks

        - Arguments:
            - tracks: np.array of shape (nb_tracks, track_max_size, 4)
        
        - Returns:
            - predictions: np.array of shape (nb_tracks, 4)
        '''
        #1. If flow module is accurate enough, use it
        #2. Otherwise do linear extrapolation
        if self._loss_balancer > 0:
            return self._flow_module(tracks)
        else:
            return self._linear_prediction(tracks)

    def update_model(self, tracks: torch.Tensor, gt: torch.Tensor):
        '''
        - Arguments:
            - tracks: torch.Tensor of shape (nb_tracks, track_max_size, 4)
            - gt: torch.Tensor of shape (nb_tracks, 4)
        '''
        preds = self._flow_module(tracks)
        loss = loss_fn(preds, gt)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        linear_preds = self._linear_prediction(tracks)
        linear_preds_loss = loss(preds, gt)
        if linear_preds_loss > loss:
            self._loss_balancer += 1
        else:
            self._loss_balancer -= 1
