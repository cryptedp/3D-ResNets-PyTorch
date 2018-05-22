import tensorflow
from tensorpack.dataflow import *
from datasets.sports1m import sports1m



class sports1m_dataflow(RNGDataFlow):
    def __init__(self, root_path, annotation_path, subset,
                 spatial_transform=None, temporal_transform=None, target_transform=None
                 ):
        
        self.dataset = sports1m(self,'/data/sammer/sports_1m/jpg/',annotation_path, subset,
                           spatial_transform=None, temporal_transform=None, target_transform=None)
        
    def get_data(self):
        #labels = randomize_labels(dataset.class_names)
        yield [self.dataset.data, self.dataset.class_names]

    def size(self):
        return self.dataset.__len__()
