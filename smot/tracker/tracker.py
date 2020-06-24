from typing import List

import torch

from smot.modeling.flow import FlowModel
from smot.tracker.track import Track

class Tracker(object):
    '''
    - max_age_thresh: The max age of a track is the number of time steps that have \
        passes since a new bounding box was associated to it. max_age_thresh is the \
        maximum nb of the max_age of a track before a track is considered to be inactive.
    - min_hits: The number of hits of a track is the number of bounding boxes that it 
        contains in its sequence. min_hits is the minimum number of hits a track must have
        to be considered valid.
    - tracks_history_length: The number of pos entries of a track to keep in memory

    '''
    def __init__(self, max_age_thresh = 7, min_hits = 3, tracks_history_length: int):
        self._max_age_thresh = max_age_thresh
        self._min_hits = min_hits
        self._tracks_history_length = tracks_history_length
        self._tracks: List[Track] = []
        self._flow_model = FlowModel()
        self._current_time_step = -1
    
    def _get_active_tracks(self) -> List[Track]:
        '''
        - Returns:
            - active_tracks: A list of tracks that are active at the current time.
        '''
        return [t for t in self._tracks if t.time_since_update(self._current_time_step) < self._max_age_thresh]
    
    def track(self, bboxes: torch.Tensor):
        """
        - Requires: 
            - This method must be called once for each frame even with empty detections.
            - The time step of the tracker increases by one each time the track method is called.

        - Definitions: 
            - A track is a sequence (in discrete time) of bounding boxes that correspond to the same \
                object.
            - A valid track is a track of size larger than min_hits.
            - An active track is a track with time_since_update less than max_age_thresh. The max age of a track \
                is the number of time steps that have passed since a new bounding box was associated to it.
        
        The algorithm:
        
        1. Predicts where current active tracks would be now, using the flow model

        2. Associates input bboxes to predictions of previous tracks if the flow model \
            prediction confidence is high, otherwise associates input bboxes to last known \
            position of a track.  Only associate bboxes to active tracks.  Unactive tracks
            can be deleted to reduce memory footprint.

        3. Use non-controversial associations to learn association parameters.

        4. Update current tracks with the associations.

        5. Pass non-controversial associations for the flow model to learn

        6. Return the input bboxes with the id of the track to which they belong. \
            Use -1 if not associated to any track.  Only assign id of track to bboxes if \
            track is valid.

        - Arguments:
            - bboxes: a torch.Tensor of detections in the format [[ymin,xmin,ymax,xmax,score],[ymin,xmin,ymax,xmax,score],...]
                
        - Returns:
            - tracks: A similar array, where the last column is the object or track id.  The number of objects returned may differ from the number of detections provided.
        """

        #1. Predict where current active tracks would be now, using the flow model
        self._current_time_step += 1
        active_tracks = self._get_active_tracks()
        active_tracks_history_l = [a.last_k_pos(self._tracks_history_length) for a in active_tracks]
        active_tracks_history_t = torch.stack(active_tracks_pos_l)  # shape is: (nb_tracks, tracks_history_length, 4)
        tracks_predictions = self._flow_model.predict(active_tracks_history_t)

        #2. Associate input bboxes to predictions.
        # TODO: Continue here.






        
        




