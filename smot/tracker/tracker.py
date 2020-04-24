import numpy as np

class Tracker(object):
    '''
    - max_age_thresh: The max age of a track is the number of time steps that have \
        passes since a new bounding box was associated to it. max_age_thresh is the \
        maximum nb of the max_age of a track before a track is considered to be inactive.
    - min_hits: The number of hits of a track is the number of bounding boxes that it 
        contains in its sequence. min_hits is the minimum number of hits a track must have
        to be considered valid.
    '''
    def __init__(self, max_age_thresh = 7, min_hits = 3):
        pass

    def track(bboxes: np.array):
        """
        - Requires: 
            - This method must be called once for each frame even with empty detections.
            - The time step of the tracker increases by one each time the track method is called.

        - Definitions: 
            - A track is a sequence (in discrete time) of bounding boxes that correspond to the same \
                object.
            - A valid track is a track of size larger than min_hits.
            - An active track is a track with max_age less than max_age_thresh. The max age of a track \
                is the number of time steps that have passed since a new bounding box was associated to it.
        
        The algorithm:
        
        1. Predicts where current active tracks would be now, using the flow model

        2. Associates input bboxes to predictions of previous tracks if the flow model \
            prediction confidence is high, otherwise associates input bboxes to last known \
            position of a track.  Only associate bboxes to active tracks.  Unactive tracks
            can be deleted to reduce memory footprint.

        3. Update current tracks with the associations.

        4. Pass non-controversial associations for the flow model to learn

        5. Return the input bboxes with the id of the track to which they belong. \
            Use -1 if not associated to any track.  Only assign id of track to bboxes if \
            track is valid.

        - Arguments:
            - bboxes: a numpy array of detections in the format [[ymin,xmin,ymax,xmax,score],[ymin,xmin,ymax,xmax,score],...]
                
        - Returns:
            - tracks: A similar array, where the last column is the object or track id.  The number of objects returned may differ from the number of detections provided.
        """
        pass