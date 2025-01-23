import faiss
import numpy as np
import pickle
import os
from typing import Dict , Any  , List
import time
import json

class FAISSManagerIVF:
    """
    Class for managing FAISS db for dynamic training and adding 

    """

    def __init__(self, n_cluster:int , embedding_dim: int = 512 , sub_vector_count:int=16, verbose=False):

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

        if len(self.text_temp) == 10000 or len(self.image_temp) == 10000:
             self.train_add()

        else:
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

        if len(self.text_temp) < self.n_cluster:
            print(f"Adjusting the clusters , current: {self.n_cluster}")
            self.n_cluster = int(len(self.text_temp) ** 0.5)

        if not self.text_index.is_trained:
            self.text_index.train(text_stack)
        self.text_index.add(text_stack)

        #to store metadata
        for i , metadata in enumerate(self.text_temp_metadata):
            faiss_id = self.text_index.ntotal - len(self.text_temp) + i
            self.text_temp_metadata[faiss_id] = metadata

        image_stack = np.vstack(self.image_temp)
        if not self.image_index.is_trained:
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

    def reset_index(self):
        """
        Function to reset index to null
        """
        self.text_index.reset()
        self.image_index.reset()

    def current_size(self) -> tuple:
        """
        Function to give current sizes of indexs
        return:
            tuple(text index size , image index size)
        """
        image_size = self.image_index.ntotal
        text_size = self.text_index.ntotal

        return (text_size , image_size)
    
    def save_state(self):
        """
        Function to save state
        """
        faiss.write_index(self.text_index , "index/text_index.index")
        faiss.write_index(self.image_index , "index/image_index.index")

        with open("index/file_meta.json" , "w+") as file:
            json.dump(self.metadata , file)

        print("saved")

    def load_state(self):
        """
        Function to load state
        """
        self.text_index = faiss.read_index("index/text_index.index")
        self.image_index = faiss.read_index("index/image_index.index")

        with open("index/file_meta.json" , "w+") as file:
            self.metadata = json.load(file)



class FAISSManagerHNSW:
    """
    Class to manage FASISS db with HNSW and PQ
    """

    def __init__(self , embedding_dim:int = 512 ,subvector_count:int = 16 , nbit:int = 8 , verbose=False):

        # HYPERPARAMS
        #emebdding dim
        self.embedding_dim = embedding_dim
        # for PQ
        self.subvector_count = subvector_count
        self.nbit = nbit
        #for HNSW
        self.M = 32

        self.text_index = faiss.IndexHNSWPQ(embedding_dim, subvector_count , nbit)
        self.image_index = faiss.IndexHNSWPQ(embedding_dim , subvector_count , nbit)

        self.text_index.hnsw.efConstruction = 80
        self.image_index.hnsw.efConstruction = 80

        self.text_metadata = {}
        self.image_metadata = {}

        self.text_temp = []
        self.image_temp = [] 
        self.text_temp_metadata = []
        self.image_temp_metdata = []

        self.verbose = verbose


    def store_temp(self,  type:str , embedding:np.array  , metadata:Dict):
        """
        Funtion to store embedding in temporary list 
        """

        if len(self.text_temp) >= 10000 or len(self.image_temp) >= 10000:
            self.train_add()

        else:

            if embedding.ndim == 1:
                embedding = embedding.reshape(1 , -1)
            
            #smoll error handeling step 
            if embedding.shape[1] != self.embedding_dim:
                raise ValueError(f"Embedding must have dim {self.embedding_dim}")
            

            if type == "image":
                self.image_temp.append(embedding.astype("float32"))
                self.image_temp_metadata.append(metadata)

            elif type == "text":
                self.text_temp.append(embedding.astype("float32"))
                self.text_temp_metadata.append(metadata)

            else:
                raise Exception("Incorrect embedding type")
            
    
    def train_add(self):

        if self.verbose:
            print("training ....")

        text_stack = np.vstack(self.text_temp)
        self.text_index.train(text_stack)

        self.text_index.add(text_stack)

        #storirng metadata

        for i , meta in enumerate(self.text_temp_metadata):
            faiss_id = self.text_index.ntotal - len(self.text_temp) + i
            self.text_metadata[faiss_id] = meta


        image_stack = np.vstack(self.image_temp)
        self.image_index.train(image_stack)
        self.image_index.add(image_stack)


        for i , meta in enumerate(self.image_temp_metdata):
            faiss_id = self.image_index.ntotal - len(self.image_temp) + i
            self.image_metadata[faiss_id] = meta

        
        self._clear_temp()

    def _clear_temp(self):
        self.text_temp = []
        self.image_temp = []
        self.text_temp_metadata = []
        self.image_temp_metadata = []

    def reset_index(self):
        """
        Function to reset index to null
        """
        self.text_index.reset()
        self.image_index.reset()

    def current_size(self) -> tuple:
        """
        Function to give current sizes of indexs
        return:
            tuple(text index size , image index size)
        """
        image_size = self.image_index.ntotal
        text_size = self.text_index.ntotal

        return (text_size , image_size)

    def save_state(self):
        """
        Function to save state
        """
        faiss.write_index(self.text_index , "index/text_index.index")
        faiss.write_index(self.image_index , "index/image_index.index")

        with open("index/file_meta.json" , "w+") as file:
            json.dump(self.metadata , file)

        print("saved")

    def load_state(self):
        """
        Function to load state
        """
        self.text_index = faiss.read_index("index/text_index.index")
        self.image_index = faiss.read_index("index/image_index.index")

        with open("index/file_meta.json" , "w+") as file:
            self.metadata = json.load(file)


        
        

    

        
         

    
    
    





    


