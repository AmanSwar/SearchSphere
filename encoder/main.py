import os
import psutil
from multiprocessing import Process , Queue
import faiss
import numpy as np
import traceback 
import argparse

#util files
import encoder.config as config
import encoder.utils as utils
import encoder.embedding as embedding


parser = argparse.ArgumentParser(description="arg parser for cli")

parser.add_argument("--verbose" , action="store_true" , help="too see which directory its currently at")

args = parser.parse_args()




#temp
search_dir = "/home/aman/code/searchsp/test/tempsearchdir"
FILE_QUEUE = Queue()
CONTENT_QUEUE = Queue()
METADATA_MAP = {}

content_extractor_func = {
    "pdf" : utils.pdf_extractor,
    "txt" : utils.text_extractor,
    "docx" : utils.docs_extractor,
    "ppt" : utils.ppt_extractor,
    "excel" : utils.excel_extractor,
    "md" : utils.markdown_extractor
}

INDEX = faiss.IndexFlatL2(512)


def traverse_all_drives():
    """
    Function to traverse all dirs in a system
    *always start from C:
    
    """
    partitions = psutil.disk_partitions()
    #list down the partitions of the drive
    for partition in partitions:
        # get specific partition
        drive = partition.mountpoint 
        # for debugging...
        print(f"Traversing {drive}...")
        for dirpath, dirnames, filenames in os.walk(drive):
            if config.RESTRICTED_DIRS_INTIAL in [folder_name[0] for folder_name in dirpath]:
                continue
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)


def test_traversal():
    try:
        for dirpath , dirnames , filenames  in os.walk(search_dir):
            
            for filename in filenames:
                file_path = os.path.join(dirpath , filename)
                if args.verbose:
                    print(f"Traversing directory : {dirpath}")
                    print(f"current file : {filename}")
                file_ext = file_path.split('.')[-1]
                if file_ext in config.SUPPORTED_EXT_IMG or file_ext in config.SUPPORTED_EXT_TEXT:
                    FILE_QUEUE.put(file_path)
                    

        FILE_QUEUE.put(None)
    
    except Exception as e:
        print(f"error in test traversal : {e}")
        traceback.print_exc()



def content_extract():
    """
    Function to extract contents from the files in queue

    """
    try:
        while True:
            file_path = FILE_QUEUE.get()
            if file_path is None:
                CONTENT_QUEUE.put(None)
                break

            file_meta_data = utils.get_meta(file_path=file_path)
            file_ext = file_path.split('.')[-1]
            # if text based files
            if file_ext in config.SUPPORTED_EXT_TEXT:
                content = content_extractor_func[file_ext](file_path=file_path)
                content_dic = {"content" : content , "metadata" : file_meta_data}
                CONTENT_QUEUE.put(content_dic)

            #if image
            if file_ext in config.SUPPORTED_EXT_IMG:
                content_dic = {"content" : file_path + "~" , "metadata" : file_meta_data}
                CONTENT_QUEUE.put(content_dic)
    
    except Exception as e:
        print(f"error in content extraction phase : {e}")
        traceback.print_exc()


def generate_and_store_embedding():
    """
    Generates embedding of contents and store it into FIASS db
    
    """
    try:
        while True:
            content_data = CONTENT_QUEUE.get()
            if content_data is None:
                break

            content = content_data["content"]
            metadata = content_data["metadata"]
            generated_embedding = None

            #handle the case of images
            if content[-1] == "~":
                img_content = content[:-1]
                #double check to avoid edge cases
                if os.path.exists(img_content):
                    '''handel images'''
                    generated_embedding = embedding.image_extract(img_content)

            else:
                '''handle texts'''
                generated_embedding = embedding.text_extract(content)

            if generated_embedding is not None:
                store(embedding=generated_embedding , metadata=metadata)

    except Exception as e:
        print(f"error while generating and storing embedding : {e}")
        traceback.print_exc()

def store(embedding , metadata:dict):
    """
    Function to add the embedding and metadata 
    Args:
        embedding (torch.tensor) : embedding vector of any content
        metadata (dict) : metadata regarding the file
    """

    global METADATA_MAP

    embedding_np = np.array([embedding] , dtype="float32")
    faiss_id = INDEX.ntotal
    INDEX.add(embedding_np)
    METADATA_MAP[faiss_id] = metadata


def terminate_process(process: Process):
    """
    function to terminate all the processes

    """

    for proc in process:
        if proc.is_alive:
            proc.terminate()
            proc.join()
    
    print("all process bye bye...")

        


if __name__ == "__main__":
    import time

    
    start_time = time.time()
    # traverse_all_drives()
    

    process = [
        Process(target=test_traversal),
        Process(target=content_extract),
        Process(target=generate_and_store_embedding)
    ]

    try:

        for proc in process:
            proc.start()

        
        for proc in process:
            proc.join()

    except Exception as e:
        print(f"exception in main process {e}")

        traceback.print_exc()
        terminate_process(process)

 
        #index reset 
        print(f"index before reset : {INDEX.ntotal}")
        INDEX.reset()
        print(f"index after reset : {INDEX.ntotal} ")

    except KeyboardInterrupt:
        print(f"keyboard interrupt")
        terminate_process(process)
        
        print(f"index before reset : {INDEX.ntotal}")
        INDEX.reset()
        print(f"index after reset : {INDEX.ntotal} ")
    


    end_time = time.time() - start_time

    print(f"done , time taken : {end_time}")



    
    

    

    



    
    
