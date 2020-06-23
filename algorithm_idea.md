# Smot

It will use self-supervised techiques to build a model of the flow of the people in the place, and to learn to make associations

## At each time step
- Predict the bounding boxes
- Predict where previous tracks would be now, using the flow model.
    - Give higher confidence to tracks with more recent observations (t - 1 has more confidence than t - 2)
- Use previous tracks predictions with current bounding boxes to associate

## Learning:
- Model of the flow of the world
- To predict a track takes as input a track A plus tracks of objects close to track A. It then analyzes how the movement of a track is affected by other tracks, in order to do the predicition.
- It learns by using for training associations from the live run that are non-controversial

## Details of the architecture:
- Step 1: For each track, compute its speed and direction using the last k steps.
- Use this information to update a 2d flow map of the world that keeps track of average spped and direction of tracks in that space.
- Step 2: For each track, take as input the k closest tracks data and build a 2d flow map of the movement of the closest tracks and of these track within a square radius, placing the current track at the center. Use a ROI layer to extract the region of interest for that track.
- Use convolutions to combine the flow maps from Step 1 and step 2 and predict the speed and direction of the track on the last step, and use this to predict position.

## Other notes:
- For tracks missing steps at the end of the track, we can do multihop predictions.