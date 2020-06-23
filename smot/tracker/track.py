import torch

class Track(object):
    '''
    #TODO: This class exists for the sake of readability of the code and simplicity of
    writing it, but at some point should be removed. The tracker should be written
    instead to do all operations over vectors instead of over this.

    - Properties:
        - hits: Number of times that the track has been updated
        - hit_streak: Current record of consecutive updates
        - age: Time since the track was created until now
        - time_since_update: How many timesteps have passed since last update
    '''
    def __init__(self, track_id: int, history_length: int):
        self._track_history: torch.Tensor = torch.full((history_length, 4), float('nan'))
        self._history_length = history_length
        self._hits = 0
        self._hit_streak = 0
        self._first_hit_step = -1
        self._history = []
        self._id = track_id
        self._last_tracker_step = -1
        self._pos = None
    
    def update(self, current_step: int, pos: torch.Tensor):
        '''
        - Arguments:
            - tracker_step (int): Contains the discrete time step to which this track belongs
            - pos (torch.Tensor): of shape (4,) [ymin, xmin, ymax, xmax]
        '''
        self._hits += 1
        if self._first_hit_step == -1:
            self._first_hit_step = current_step
        if self._last_tracker_step == current_step - 1:
            self._hit_streak += 1
        self._last_tracker_step = current_step
        idx = (current_step - self._first_hit_step) % self._history_length
        self._track_history[idx] = pos

    @property
    def track_id(self):
        return self._id
    
    @property
    def current_pos(self) -> torch.Tensor:
        '''
        - Returns:
            - torch.Tensor of size (4,)
        '''
        idx = (self._last_tracker_step - self._first_hit_step) % self._history_length
        return self._track_history[idx]
    
    def last_k_pos(self, k: int) -> torch.Tensor:
        '''
        - Returns:
            - torch.Tensor of size (k, 4)
        '''
        # TODO
        pass

    @property
    def hits(self):
        return self._hits
    
    @property
    def age(self):
        return self._last_tracker_step - self._first_hit_step
    
    @property
    def hit_streak(self):
        return self._hit_streak
    
    @property
    def time_since_update(self, current_time_step: int):
        return current_time_step - self._last_tracker_step
