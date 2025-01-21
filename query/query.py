import faiss
import numpy as np
import json
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from encoder.embedding import text_extract

INDEX = faiss.read_index('index/faiss_index.index')
with open('index/file_meta.json' , 'r') as file:
    METADATA_MAP = json.load(file)

parser = argparse.ArgumentParser(description="arg parser for cli")
parser.add_argument("--search" , type=str ,required=True , help="your query")
args = parser.parse_args()


query = args.search


def query_to_embedding(query:str):
    """
    Converts the query to embedding
    args:
        query (str) : the query to be searched
    """
    
    query_embed = text_extract(query)
    return query_embed


def query_faiss_and_meta(query_embedding):

    k = 5
    query_embedding = np.array(query_embedding , dtype="float32").reshape(1 , -1)
    dist , indices = INDEX.search(query_embedding , k)

    results = []
    print("indices : " , indices)
    for i in range(len(indices[0])):
        faiss_id = indices[0][i]
        result_metadata = METADATA_MAP[str(faiss_id)]
        print(result_metadata["file_name"])

    return results


def query_faiss_debug(query_embedding):
    k = 10
    query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)
    
    dist, indices = INDEX.search(query_embedding, k)

    image_results = []
    text_results = []

    for i in range(len(indices[0])):
        faiss_id = str(indices[0][i]) 
        
        if faiss_id in METADATA_MAP:
            metadata = METADATA_MAP[faiss_id]
            file_type = metadata['file_type'].lower()
            
            result = {
                'rank': i + 1,
                'file': metadata['file_name'],
                'distance': dist[0][i],
                'type': file_type
            }
            
            if file_type in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                image_results.append(result)
            else:
                text_results.append(result)
    
    print("\nImage Results:")
    if image_results:
        for res in image_results:
            print(f"Rank {res['rank']}: {res['file']} (distance: {res['distance']:.4f})")
    else:
        print("No image results found!")
    
    print("\nText Results:")
    if text_results:
        for res in text_results:
            print(f"Rank {res['rank']}: {res['file']} (distance: {res['distance']:.4f})")
            
    return image_results, text_results


def query_img_boost(query_embedding):
    k = 10
    query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)
    
    similarities, indices = INDEX.search(query_embedding, k)
    
    # Store all results with boosted image similarities
    results = []
    for i in range(len(indices[0])):
        faiss_id = str(indices[0][i])
        if faiss_id in METADATA_MAP:
            metadata = METADATA_MAP[faiss_id]
            file_type = metadata['file_type'].lower()
            similarity = similarities[0][i]
            
            # Apply boost to image results
            if file_type in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                similarity *= 5  # Boost image similarities by 50%
            
            results.append({
                'file': metadata['file_name'],
                'type': file_type,
                'similarity': similarity,
                'is_image': file_type in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
            })
    
    # Sort by boosted similarity
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Print results
    print("\nResults (higher similarity is better):")
    for i, res in enumerate(results):
        print(f"Rank {i+1}: {res['file']} ({res['type']}) - similarity: {res['similarity']:.4f}")
    
    return results

if __name__ == "__main__":

    qu_embed = query_to_embedding(query)

    query_img_boost(qu_embed)

    # print(METADATA_MAP)

