import os
import psutil
from multiprocessing import Process , Queue , Manager

import numpy as np
import traceback 
import argparse
import torch
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import json

import encoder.config as config
import encoder.utils as utils
import encoder.embedding as embedding
from encoder.faiss_base import FAISSManagerHNSW

parser = argparse.ArgumentParser(description="arg parser for cli")
parser.add_argument("--verbose" , action="store_true" , help="too see which directory its currently at")
parser.add_argument("--dir" , type=str, help="Dir to create embeddings")
args = parser.parse_args()


search_dir = args.dir if args.dir is not None else "/home/aman/code/searchsp/test/tempsearchdir"

content_extractor_func = {
    "pdf" : utils.pdf_extractor,
    "txt" : utils.text_extractor,
    "docx" : utils.docs_extractor,
    "ppt" : utils.ppt_extractor,
    "excel" : utils.excel_extractor,
    "md" : utils.markdown_extractor
}

faiss_manager = FAISSManagerHNSW(verbose=args.verbose)

def test_traversal():       
    faiss_manager.reset_index()
    current_size = faiss_manager.current_size()
    print(f"current item in text index : {current_size[0]}")
    print(f"current item in image index : {current_size[1]}")
    try:
        for dirpath , dirnames , filenames  in tqdm(os.walk(search_dir)):
            for filename in filenames:
                file_path = os.path.join(dirpath , filename)

                #verbose
                if args.verbose:
                    print(f"Traversing directory : {dirpath}")
                    print(f"current file : {filename}")

                file_ext = file_path.split('.')[-1]
                if file_ext in config.SUPPORTED_EXT_IMG or file_ext in config.SUPPORTED_EXT_TEXT:
                    content_extract(file_path=file_path)
        
        #calling train after traversal ends
        faiss_manager.train_add()
    except Exception as e:
        print(f"error in test traversal : {e}")
        traceback.print_exc()


def content_extract(file_path):
    """
    Function to extract contents from the files in queue

    """
    try:
        #verbose
        if args.verbose:
            print(f"current file getting extracted : {file_path}")

        # get meta data of the file -> dict
        file_meta_data = utils.get_meta(file_path=file_path)
        file_ext = file_path.split('.')[-1]
        # if text based files
        if file_ext in config.SUPPORTED_EXT_TEXT:
            
            content = content_extractor_func[file_ext](file_path=file_path)
            content_dic = {"content" : content , "metadata" : file_meta_data}
            generate_embedding(content_dic)

            if args.verbose:
                print(f"extracted file {file_path}")


        #if image
        if file_ext in config.SUPPORTED_EXT_IMG:
            content_dic = {"content" : file_path + "~" , "metadata" : file_meta_data}
            generate_embedding(content_dic)
            if args.verbose:
                print(f"Successfully processed img {file_path}")

            
    
    except Exception as e:
        print(f"error in content extraction phase : {e}")
        traceback.print_exc()


def generate_embedding(content_data):
    """
    Generates embedding of contents and store it into FIASS db
    
    """
    try:  
        content = content_data["content"]
        metadata = content_data["metadata"]
        generated_embedding = None
        embed_type = None
	 	
        #handle the case of images
        if content[-1] == "~":
            img_content = content[:-1]
            #double check to avoid edge cases
            if os.path.exists(img_content):
                '''handel images''' 
                if args.verbose:
                    print("generating embedding ...")
                embed_type = "image"
                generated_embedding = embedding.image_extract(img_content)

        else:
            '''handle texts'''
            embed_type = "text"
            generated_embedding = embedding.text_extract(content)

        if generated_embedding is not None:
            #ensuring its a numpy array
            if args.verbose:
                print("generated")
            if isinstance(generated_embedding ,torch.Tensor):
                generated_embedding = generated_embedding.cpu().numpy()
            data = (embed_type , generated_embedding , metadata)
            store_embedding(data)
            
    except Exception as e:
        print(f"error while generating and storing embedding : {e}")
        traceback.print_exc()



def store_embedding(data:tuple):
    """
    Function to add the embedding and metadata 
    Args:
        embedding (torch.tensor) : embedding vector of any content
        metadata (dict) : metadata regarding the file
    """

    try:
        embed_type, embedding_vec , metadata = data

        #for debugging purposes
        norm = np.linalg.norm(embedding_vec)
        print(f"Embedding norm for {metadata['file_name']}: {norm}")
        faiss_manager.store_temp(type=embed_type , embedding=embedding_vec , metadata=metadata)
        

    except Exception as e:
        print(f"Error which storing embedding : {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import time
    start_time = time.time()


    test_traversal()
    print(f"TOTAL ENTERIES : {faiss_manager.current_size()}")
    faiss_manager.save_state()
    end_time = time.time() - start_time

    print(end_time)