import openvino.runtime as ov
import numpy as np
import cv2
import random
from scipy.spatial.distance import cdist
from models import ReidModel, YoLoV5Model

class ClusterFeature:
    """
    Store 4 embedding features of a Track object
    """
    def __init__(self, feature_len, initial_feature=None):
        
        # initialize the variables
        self.clusters = []
        self.clusters_sizes = []
        self.feature_len = feature_len
        
        if initial_feature is not None:

            self.clusters.append(initial_feature)
            self.clusters_sizes.append(1)

    def get_clusters_matrix(self):
        return np.array(self.clusters).reshape(len(self.clusters), -1)

    def __len__(self):
        return len(self.clusters)


    def update(self, feature_vec):
        # update the feature clusters

        # if there are less clusters than feature_len, append the feature_vec to the clusters list
        if len(self.clusters) < self.feature_len:

            self.clusters.append(feature_vec)
            self.clusters_sizes.append(1)

        # if the total clusters_sizes is less than 2*feature_len, 
        # random an index and update the vector according to the feature_vec
        elif sum(self.clusters_sizes) < 2*self.feature_len:

            idx = random.randint(0, self.feature_len - 1) # nosec - disable B311:random check
            
            self.clusters_sizes[idx] += 1
            self.clusters[idx] += (feature_vec - self.clusters[idx]) / self.clusters_sizes[idx]

        # if the total clusters_sizes is greater than 2*feature_len, 
        # compute the distance of the feature_vec and other clusters, update the cluster with the lowest cosine distance
        else:
            distances = cdist(feature_vec.reshape(1, -1),
                              np.array(self.clusters).reshape(len(self.clusters), -1), 'cosine')

            nearest_idx = np.argmin(distances)

            self.clusters_sizes[nearest_idx] += 1
            self.clusters[nearest_idx] += (feature_vec - self.clusters[nearest_idx]) / self.clusters_sizes[nearest_idx]



class Track:
    """
    Store the index, time, num_clusters of a detection
    """
    def __init__(self, index, time, num_clusters, cam_id=0, feature=None):
        self.index = index
        self.cam_id = cam_id
        self.f_clust = ClusterFeature(num_clusters)
        self.timestamps = [time]
        if feature is not None:
            self.f_clust.update(feature)
    
    def get_features(self):
        return self.features

    def get_end_time(self):
        return self.timestamps[-1]

    def get_start_time(self):
        return self.timestamps[0]

    def __len__(self):
        return len(self.timestamps)
    
    def update(self, time, feature):
        self.timestamps.append(time)
        self.f_clust.update(feature)

            
if __name__ == "__main__":
    core = ov.Core()
    reid_model = ReidModel(core, "..\models\intel\person-reidentification-retail-0287\FP16-INT8\person-reidentification-retail-0287.xml")

    yolo_model = YoLoV5Model(core, "..\models\yolo_openvino_model\yolov5s.xml")

    image = cv2.imread(filename="..\\5981885-a-group-of-young-people-walking-down-a-street-in-a-large-city.jpg")
    result = yolo_model.forward(image)

    embedding_matrix = reid_model.forward(image, result)

    number_track = embedding_matrix.shape[0]

    for i in range(number_track):
        if i == 0:
            track = Track(0, i, 4, feature=embedding_matrix[i])
        else:
            track.update(i, embedding_matrix[i])
    
    print(len(track.f_clust.clusters))
    print(track.f_clust.clusters_sizes)