from transformers import AutoTokenizer , MobileBertForSequenceClassification
import torch
from tqdm import tqdm
import threading
import time

save_directory = "query/results/saved_model"
# Load the saved model and tokenizer
model = MobileBertForSequenceClassification.from_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model.eval()

labeltoid = {0 : "TEXT" , 1 : "IMAGE"}

def index_token(query:str):
    """
    Query function which utilize mobilebert for generating token

    args:
        query (str) => user query in normal format
    
    returns:
        predicted token -> TEXT / IMAGE
    """

    inp = tokenizer(text=query,
                    truncation=True , 
                    padding="max_length" , 
                    max_length=128 , 
                    return_tensors="pt"
                    )
    
    with torch.no_grad():
        out = model(**inp)

    logits = out.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_label = torch.argmax(probs, dim=1).item()

    return labeltoid[predicted_label]


def progress_bar(func , *args , **kwargs):

    stop_event = threading.Event()

    def bar():
        with tqdm(total=100 , unit="%" , ncols=100 , desc="Searching ...") as pbar:
            while not stop_event.is_set():
                for _ in range(10):
                    time.sleep(0.1)
                    pbar.update(1)

                    if stop_event.is_set():
                        break

            pbar.n = pbar.total
            pbar.close()


    thread = threading.Thread(target=bar)
    thread.start()

    try:
        result = func(*args , **kwargs)
        return result
    
    finally:
        stop_event.set()
        thread.join()


    


    

    