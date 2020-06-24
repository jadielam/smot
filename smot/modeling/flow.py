import torch
from torch import nn
from fvcore.nn import giou_loss, smooth_l1_loss

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
    #return smooth_l1_loss(preds, gt, 0.05)
    return giou_loss(preds, gt)

class FlowModel(object):
    '''
    - Arguments:
        - lr: learning rate for FlowModule
        - adaptive_window_size: Number of steps from the past to keep
        in order to 
    '''
    def __init__(self, lr = 0.001, adaptive_window_size = 100):
        self._flow_module = FlowModule()
        self._optimizer = torch.optim.Adam(self._flow_module.parameters(), lr = lr)
        
        # Code to be used in determining the balance of
        self._adaptive_window_size = adaptive_window_size
        self._last_k_balancers = torch.zeros((self._adaptive_window_size,))
        self._balancer_idx = 0
        self._loss_balancer_aggr = 0
        
    def _linear_prediction(self, tracks: torch.Tensor):
        '''
        Implements a weighted average that gives higher weight
        to more recent changes in location.

        - Arguments:
            - tracks: torch.Tensor of shape (nb_tracks, track_max_size, 4)
            The last entry in axis 1 (the second axis) represents the most recent
            position
        
        - Returns:
            - preds: (nb_tracks, 4)
        '''
        #1. Calculate location changes from one step to the next
        rolled_tracks = torch.roll(tracks, 1, 1)
        diff = tracks[:, 1:, :] - rolled_tracks[:, 1:, :]

        #2. Calculate current direction weighted average by weight-averaging calculations
        # in 1
        diff[torch.isnan(diff)] = 0
        weights = torch.arange(diff.shape[1], dtype = diff.dtype)
        mean_diff = torch.einsum('ijk,j->ik', diff, weights) / torch.sum(weights)  # shape: (nb_tracks, 4)

        #3. For each track, find the number of steps between latest track with non-nan values
        #and now
        # shape (nb_tracks,)
        last_index = (~torch.isnan(tracks[:,:,0])).double().argmax(dim = 1)
        nb_steps = tracks.shape[1] - 1 - last_index
        
        #4. Multiply #2 by #3.
        values_to_add = torch.einsum('ij,i->ij')


        #5. Add #4 to latest track of non-nan values and return.

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
        if self._loss_balance_aggr > 0:
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
        
        # This code below is similar to running a running window online mean
        # calculation
        if linear_preds_loss > loss:
            balancer_value = 1
        else:
            balancer_value = -1
        next_idx = (self._balancer_idx + 1) % self._adaptive_window_size
        self._loss_balancer_aggr += balancer_value
        self._loss_balancer_aggr -= self._last_k_balancers[next_idx]
        self._last_k_balancers[next_idx] = balancer_value
        self._balancer_idx = next_idx
