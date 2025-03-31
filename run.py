import encoder.main_seq
import encoder.utils
from query import query
import encoder
from encoder.main_seq import dir_traversal

import os
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  


# init

search_dir = ""


while True:
    if search_dir == "":
        print("enter search directory : ")
        temp_search_dir = input()

        search_dir = encoder.utils.prep_dir(temp_search_dir)

        if not os.path.exists(search_dir):
            print(search_dir)
            print("invalid please try again")

        else:
            break


print("GENERATING EMBEDDING")
start_time = time.time()
dir_traversal(search_dir=search_dir)
end_time = time.time() - start_time

print(f"Generated embeddings in {end_time:2f} secs")
from query import query
while True:
    print("\nEnter query : ...")
    q = input()

    query.search(q)


