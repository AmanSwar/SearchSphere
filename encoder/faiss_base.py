import faiss
import numpy as np
import pickle
import os
from typing import Dict , Any  , List


class FAISSIndexManager:

    def __init__(self ,
                dim:int ,
                index_path:str,
                metadata_path:str,
                ):
        
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path

        self.metadata_dict: Dict[int , Dict[str , Any]] = {}
        self.current_id = 0


        self.quantizer = faiss.IndexFlatL2(dim)
        
        #currently set to arbitary value
        nlist = 20
        self.index = faiss.IndexIVFFlat(self.quantizer , dim , nlist , faiss.METRIC_L2)

        self.is_trained = False


    def train_index(self , training_vectors: np.ndarray):

        if not self.is_trained:
            self.index.train(training_vectors)
            self.is_trained = True