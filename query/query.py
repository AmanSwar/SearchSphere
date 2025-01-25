import faiss
import numpy as np
import json
import argparse
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


from encoder.embedding import text_extract
from encoder.faiss_base import FAISSManagerHNSW
from query import utils


parser = argparse.ArgumentParser(description="arg parser for cli")
parser.add_argument("--search" , type=str ,required=False , help="your query")
parser.add_argument("--verbose" , action="store_true" , help="verbosify the output")
args = parser.parse_args()


user_query = args.search

#faiss init
faiss_manager = FAISSManagerHNSW(verbose=args.verbose)
faiss_manager.load_state()

def query_extractor(query:str):
    """
    Converts the query to embedding
    args:
        query (str) : the query to be searched

    returns:
        tuple: type token , query embedding
    """
    
    query_embed = text_extract(query)
    type_token = utils.index_token(query=query)

    return (type_token ,query_embed)


def search(query:str):
    """
    main function for search

    args:
        query (str)


    """
    start_time = time.time()
    type_token , query_embed = query_extractor(query=query)

    if type_token == "TEXT":
        dist , indice , metadata = utils.progress_bar(faiss_manager.search_text , query_embed=query_embed)

    elif type_token == "IMAGE":
        dist , indice , metadata = utils.progress_bar(faiss_manager.search_image ,query_embed=query_embed)
    else:
        raise Exception("Invalid token generated ...")

    end_time = time.time() - start_time
    if args.verbose:
        print(metadata)
        print(indice)

    for i in range(len(indice)):
        faiss_id = indice[i]
        result_meta = metadata[str(faiss_id)]
        

        if args.verbose:
            
            print("\n")
            print(f"file name -> {result_meta["file_name"]}")
            print(f"similarity score -> {dist[i]}")
            print(f"File Info...")
            print(f"file location -> {result_meta["file_path"]}")
        else:
            print("\n")
            print(result_meta["file_name"])
            print(f"file location -> {result_meta["file_path"]}")
            print(f"searched in {end_time:3f} secs")




if __name__ == "__main__":

    search(user_query)

