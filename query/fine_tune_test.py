from transformers import AutoTokenizer, MobileBertForSequenceClassification
import pandas as pd
import torch

save_directory = "/home/aman/code/searchsp/query/results/saved_model"
# Load the saved model and tokenizer
model = MobileBertForSequenceClassification.from_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)

print("Model and tokenizer loaded successfully!")

model.eval()



labeltoid = {0 : "TEXT" , 1 : "IMAGE"}
df = pd.read_csv("/home/aman/code/searchsp/query/data/multimodal_search_dataset.csv")
import time

start_time = time.time()
for i in range(5):
    text = df['query'][i]

    inp = tokenizer(text=text , truncation=True , padding="max_length" , max_length=128 , return_tensors="pt")

    with torch.no_grad():
        out = model(**inp)

    logits = out.logits

    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get the predicted label
    predicted_label = torch.argmax(probs, dim=1).item()

    print(f"Predicted label: {predicted_label}")
    print(f"Probabilities: {probs.tolist()}")

    if labeltoid[predicted_label] == df['label'][i]:
        print("True")

    else:
        print("false")

end_time = time.time() - start_time

print(end_time)