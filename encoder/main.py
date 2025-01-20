import os
import psutil
from multiprocessing import Process , Queue , Manager
import faiss
import numpy as np
import traceback 
import argparse
import torch
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#util files
import encoder.config as config
import encoder.utils as utils
import encoder.embedding as embedding


parser = argparse.ArgumentParser(description="arg parser for cli")
parser.add_argument("--verbose" , action="store_true" , help="too see which directory its currently at")
args = parser.parse_args()


#temp
search_dir = "/home/aman/code/searchsp/test/tempsearchdir"
content_extractor_func = {
    "pdf" : utils.pdf_extractor,
    "txt" : utils.text_extractor,
    "docx" : utils.docs_extractor,
    "ppt" : utils.ppt_extractor,
    "excel" : utils.excel_extractor,
    "md" : utils.markdown_extractor
}





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


def main():
    with Manager() as manager:
        FILE_QUEUE = Queue()
        CONTENT_QUEUE = Queue()
        EMBEDDING_QUEUE = Queue()
        METADATA_MAP = manager.dict() # in shared memory
        INDEX = faiss.IndexFlatL2(512)

        def test_traversal():
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
                file_processed = 0
                while True:
                    try:
                        file_path = FILE_QUEUE.get(timeout=60)
                        if file_path is None:
                            CONTENT_QUEUE.put(None)
                            break
                        #verbose
                        if args.verbose:
                            print(f"current file getting extracted : {file_path}")

                        file_meta_data = utils.get_meta(file_path=file_path)
                        file_ext = file_path.split('.')[-1]
                        # if text based files
                        if file_ext in config.SUPPORTED_EXT_TEXT:

                            try:
                                content = content_extractor_func[file_ext](file_path=file_path)
                                content_dic = {"content" : content , "metadata" : file_meta_data}
                                CONTENT_QUEUE.put(content_dic)
                                file_processed += 1

                                if args.verbose:
                                    print(f"extracted file {file_path}")

                            except Exception as e:
                                print(f"Error extracting content from {file_path}")
                                continue

                        #if image
                        if file_ext in config.SUPPORTED_EXT_IMG:
                            content_dic = {"content" : file_path + "~" , "metadata" : file_meta_data}
                            CONTENT_QUEUE.put(content_dic)
                            file_processed += 1

                            if args.verbose:
                                print(f"Successfully processed img {file_path}")

                    except Queue.empty:
                        print("content extraction timed out - no new files")
                        CONTENT_QUEUE.put(None)
                        break
            
            except Exception as e:
                print(f"error in content extraction phase : {e}")
                traceback.print_exc()
                CONTENT_QUEUE.put(None)


        def generate_embedding():
            """
            Generates embedding of contents and store it into FIASS db
            
            """
            try:
                while True:
                    try:
                        content_data = CONTENT_QUEUE.get(timeout=60)
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
                            #ensuring its a numpy array
                            if isinstance(generated_embedding ,torch.Tensor):
                                generated_embedding = generated_embedding.cpu().numpy()
                            # store(embedding=generated_embedding , metadata=metadata)
                            EMBEDDING_QUEUE.put((generated_embedding , metadata))
                    
                    except Queue.empty:
                        print("embedding generation timed out")
                        EMBEDDING_QUEUE.put(None)
                        break

            except Exception as e:
                print(f"error while generating and storing embedding : {e}")
                traceback.print_exc()
                EMBEDDING_QUEUE.put(None)

        def store_embedding():
            """
            Function to add the embedding and metadata 
            Args:
                embedding (torch.tensor) : embedding vector of any content
                metadata (dict) : metadata regarding the file
            """

            try:
                while True:
                    data = EMBEDDING_QUEUE.get()

                    if data is None:
                        break
                        
                    embedding_vec , metadata = data

                    if embedding_vec.ndim == 1:
                        embedding_np = embedding_vec.reshape(1 , -1)
                    else:
                        embedding_np = embedding_vec

                embedding_np = embedding_np.astype('float32')
                faiss_id = INDEX.ntotal
                INDEX.add(embedding_np)
                METADATA_MAP[faiss_id] = metadata


                if args.verbose:
                    print(f"Added Embedding {faiss_id} to index")

            except Exception as e:
                print(f"Error which storing embedding : {e}")
                traceback.print_exc()

        def terminate_process(process: Process):
            """
            function to terminate all the processes

            """

            for proc in process:
                if proc.is_alive:
                    proc.terminate()
                    proc.join()
            
            print("all process bye bye...")

 

        process = [
            Process(target=test_traversal),
            Process(target=content_extract),
            Process(target=generate_embedding),
            Process(target=store_embedding)
        ]

        try:

            for proc in process:
                proc.start()

            if args.verbose:
                iter_ = 0
                while any(p.is_alive() for p in process):
                    print(f"\niter : {iter_},"
                        f"File queue size -> {FILE_QUEUE.qsize()},"
                        f"Content -> {CONTENT_QUEUE.qsize()},"
                        f"Embedding -> {EMBEDDING_QUEUE.qsize()}," 
                        f"db size -> {INDEX.ntotal}" ,end="")
                    
                    time.sleep(1)
                    iter_ += 1

            
            for proc in process:
                proc.join(timeout=60)

            print(f"Final index size: {INDEX.ntotal}")
            print(f"Final metadata map size: {len(METADATA_MAP)}")
            
            return INDEX, dict(METADATA_MAP)
        
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
            


            



if __name__ == "__main__":

    import time
    start_time = time.time()

    index, metadata_map  = main()

    end_time = time.time() - start_time

    print(f"Done, time taken: {end_time}")
    print(f"Current items in FAISS: {index.ntotal}")

    
    

    

    



    
    
