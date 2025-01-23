from transformers import AutoTokenizer , MobileBertForSequenceClassification
import torch


save_directory = "/query/results/saved_model"
# Load the saved model and tokenizer
model = MobileBertForSequenceClassification.from_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model.eval()

labeltoid = {0 : "TEXT" , 1 : "IMAGE"}

def index_token(query:str):
    """
    Query function which utilize mobilebert for generating token
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


    

    

    