# SEARCH SPHERE

![SEARCH SPHERE LOGO](https://github.com/AmanSwar/SearchSphere/blob/master/image.png)
  
### **OVERVIEW**

**Windows Search bar sucks** , you can't search any file until and unless you know the exact name of the file which often we don't .

Hence **Search sphere**

Its a multimodal search engine meaning you can search for both images and documents of kinds using natural language

Current Supported file types:
- Textual based
	- Word documents
	- Pdf
	- Power point presentations
	- Text files
	- markdown files
- Image based 
	- jpeg
	- png
	- jpg


### HOW TO RUN

First of all use 
```
pip install -r requirements.txt
```

to install all the requirements then

just run ``` run.py``` file using
```
python -m run
```

thats it !!!


### HOW IT WORKS

1) First it traverse the whole directory and its sub directories which the user gives and picks all the supported files
2) Then it generate embeddings of not the whole file but up to some meaningful extent (see the logic in ``` encoder/utils.py``` file)
3) Now according to the file type , the embedding is stored in 2 different FAISS database one for text and one for image (reason of 2 different database ? well using VLMs like MobileCLIP i don't know why but it tends to have more tendency to return textual answer if the prompt is in text)
4) This concludes the embedding generation , time for query
5) When the user type a query first its passed through MobileBERT model which is custom trained (weights included in ```query/result/save_model```) to output 2 tokens :- IMAGE and TEXT which tells whether the user is asking for an image or a textual content . This helps to reduce the search space and provide better result
6) Now according to token , we search in the particular database . In FAISS we are using HNSWFlat search (reason for this ? FlatL2 takes alot of time , IVF needs training of embedding , since our embedding is generated in runtime we cannot train it properly , PQ also needs training)
7) The top 5 files with highest similarity score is returned along with file location . Enjoyy !!


### FUTURE SCOPES AND DIRECTION

1) add more file support + video support 
2) add small LLM which can make this whole thing like a offline perplexity 
3) better CLI
