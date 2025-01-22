import faiss
import numpy as np
import pickle
import os
from typing import Dict , Any  , List
import time

class FAISSManager:
    """
    - stored embedding -> temp 
    - training
    - adding
        - image index
        - text index
    - quantization

    """

    def __init__(self , embedding_dim: int = 512 , n_cluster:int = 100, sub_vector_count:int=16, verbose=False):

        self.verbose = verbose

        self.subvector = sub_vector_count
        self.n_cluster = n_cluster
        self.embedding_dim = embedding_dim

        self.text_index = faiss.index_factory(self.embedding_dim , f"IVF{n_cluster},PQ{sub_vector_count}")
        self.image_index = faiss.index_factory(self.embedding_dim , f"IVF{n_cluster},PQ{sub_vector_count}")
        self.metadata = {}
        
        self.text_temp = []
        self.image_temp = []
        self.text_temp_metadata = []
        self.image_temp_metadata = []


    def store_temp(self ,type:str, embedding:np.array , metdata:Dict):
        """
        Function to store the embedding temp in a list
        args:
            embedding (torch.tensor) => embeddings
            type (str) => of 2 types
                - image
                - text
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1 , -1)
        
        #smoll error handeling step 
        if embedding.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding must have dim {self.embedding_dim}")
        
        if type == "image":
            self.image_temp.append(embedding.astype("float32"))
            self.image_temp_metadata.append(metdata)

        elif type == "text":
            self.text_temp.append(embedding.astype("float32"))
            self.text_temp_metadata.append(metdata)

        else:
            raise Exception("Incorrect embedding type")
        
    
    def train_add(self):
        """
        Function to train on new embedding
        """
        if self.verbose:
                    print("training...")


        text_stack = np.vstack(self.text_temp)
        self.text_index.train(text_stack)
        self.text_index.add(text_stack)

        #to store metadata
        for i , metadata in enumerate(self.text_temp_metadata):
            faiss_id = self.text_index.ntotal - len(self.text_temp) + i
            self.text_temp_metadata[faiss_id] = metadata

        image_stack = np.vstack(self.image_temp)
        self.image_index.train(image_stack)
        self.image_index.add(image_stack)

        for i , metadata in enumerate(self.image_temp_metadata):
             faiss_id = self.image_index.ntotal - len(self.image_temp) + i
             self.image_temp_metadata[faiss_id] = metadata


        if self.verbose:
            print("finish training...")

        self._clear_temp()
        

    def _clear_temp(self):
         self.text_temp = []
         self.image_temp = []
         self.text_temp_metadata = []
         self.image_temp_metadata = []

    
    





    


