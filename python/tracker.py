import openvino.runtime as ov
import numpy as np
import cv2

from models import ReidModel, YoLoV5Model
from track import Track, ClusterFeature


class SingleCameraTracker:
    """
    Tracking the Track object, it can tracks 15 people at the same time, and keeping 30 person in the history
    """
    def __init__(self, reid_model, MAX_NUM_TRACK=30, NUM_CLUSTERS=4, TRACKING_THRESHOLD=0.75, MIN_TRACK_LENGTH=5):

        self.reid_model = reid_model

        self.NUM_CLUSTERS = NUM_CLUSTERS 
        self.MAX_NUM_TRACK = MAX_NUM_TRACK
        self.TRACKING_THRESHOLD = TRACKING_THRESHOLD

        self.MIN_TRACK_LENGTH = MIN_TRACK_LENGTH

        self.embeddings_history = np.zeros((self.MAX_NUM_TRACK * self.NUM_CLUSTERS, 256), dtype=np.float32)  # (60, 256)

        self.tracks_history = []
        self.availability = [True] * self.MAX_NUM_TRACK

        self.time = 0

    def cosine(self, new_embedding, current_embedding):
        # calculate the length of each vector
        norm_new_embedding = np.expand_dims(np.linalg.norm(new_embedding, axis=1), -1)
        norm_current_embedding = np.expand_dims(np.linalg.norm(current_embedding, axis=1), -1)
        
        # calculate the dot product
        a = new_embedding @ current_embedding.T

        return new_embedding @ current_embedding.T / (norm_new_embedding @ norm_current_embedding.T + np.finfo(np.float32).eps)

    def process(self, frame, detections):

        # initialize the embedding matrix
        # embedding_features = [None]*len(detections)

        # get the embedding of the matrix
        
        if self.reid_model:
            # the embedding_features is a matrix containing all the the embedding of the people with shape: (..., 256)
            embedding_features = self.reid_model.forward(frame, detections)

        # execute the tracking process, i.e. giving the detected people the correct ID

        updated_index = self.tracking(embedding_features)


        self.time += 1

        return updated_index


    def tracking(self, embedding_features):
        # compute the cosine similarity matrix
        # each row represents a new embedding
        # each column represents a current embedding
        cosine_similarities = self.cosine(embedding_features, self.embeddings_history) # (..., 120)

        # store the max_value of similarity of each new embedding
        max_value = np.max(cosine_similarities, axis=1)

        # store the max_index of similarity of each new embedding
        max_index = np.argmax(cosine_similarities, axis=1)

        # initialise the updated_index list
        updated_index = []

        # loop through the max_index and update the system variables, including embeddings_history and tracks_history
        for i in range(len(max_index)):
            # check for the similarity condition
            if max_value[i] <= self.TRACKING_THRESHOLD:
                
                if True in self.availability:
                    # get the inject index 
                    inject_index = self.availability.index(True)
                else:
                    inject_index = self.find_oldest_track()
                # create a new track object
                track = Track(inject_index, 0, 4, feature=embedding_features[i])

                # update the tracks_history
                self.tracks_history.append(track)

            else:
                # get the track_index to get to the correct track object
                inject_index = max_index[i] // 4

                # get the track object
                track = self.tracks_history[inject_index]

                # update the track object
                track.f_clust.update(embedding_features[i])

            
            # get the stop_index from the current length of the clusters
            stop_index = len(track.f_clust.clusters)
            
            # update the availability list
            self.availability[inject_index] = False

            # update the embedding matrix
            self.embeddings_history[inject_index*4: inject_index*4+stop_index] = np.array(track.f_clust.clusters)
            
            # append the updated_index to return later
            updated_index.append(inject_index)

        return updated_index


    def find_oldest_track(self):
        current_time = self.time

        for track in self.tracks_history:
            age = track.timestamps[-1]
            if age < current_time:
                current_time = age
                inject_index = track.index
        
        return inject_index

