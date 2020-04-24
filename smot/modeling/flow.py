import numpy as np

class FlowModel(object):
    def __init__(self):
        pass

    def predict(tracks: np.array):
        '''
        - Arguments:
            - tracks: np.array of shape (nb_tracks, track_max_size, 5)
        
        - Returns:
            - predictions: np.array of shape (nb_tracks,)
        '''
        pass

    def update_model(tracks: np.array, preds: np.array):
        '''
        - Arguments:
            - tracks: np.array of shape (nb_tracks, track_max_size, 5)
            - preds: np.array of shape (nb_tracks,)
        '''
        pass