# Search Sphere

![SEARCH SPHERE LOGO](https://github.com/AmanSwar/SearchSphere/blob/master/image.png)

### **OVERVIEW**

**Search Sphere** is a sophisticated, local-first, multimodal search engine designed to revolutionize file discovery on your machine. Tired of traditional search tools that rely on exact filenames? Search Sphere allows you to find files—including documents and images—using natural language queries, just like you would on a web search engine.

This tool is built for anyone who needs to find files based on their *content* rather than just their names.

**Current Supported File Types:**
- **Textual:** `.pdf`, `.docx`, `.doc`, `.txt`, `.md`
- **Image:** `.jpeg`, `.jpg`, `.png`

---

### **HOW TO RUN**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AmanSwar/SearchSphere.git
    cd SearchSphere
    ```

2.  **Install dependencies:**
    This project uses several packages, including a specific version of `mobileclip` installed directly from GitHub.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    Execute the main `run.py` script. The application will guide you through the rest.
    ```bash
    python run.py
    ```
    - You will first be prompted to enter a directory to index.
    - The engine will then process the files and build the search index.
    - Once indexing is complete, you can start entering queries.

---

### **HOW IT WORKS**

Search Sphere operates in two main phases: **Indexing** and **Querying**.

**1. Indexing Phase (Embedding Generation):**
- **File Traversal:** The tool scans the user-provided directory, identifying all supported text and image files.
- **Content Extraction:** It extracts raw text from documents and identifies image paths.
- **Multimodal Embeddings:** Using Apple's **MobileCLIP** model, it converts both the extracted text and the images into 512-dimensional vector embeddings. These embeddings represent the semantic meaning of the content.
- **FAISS Indexing:** The generated embeddings are stored in two separate **FAISS (HNSWFlat)** vector indexes: one for text and one for images. This separation allows for more precise search results. Metadata for each file (like its name and path) is stored alongside the index.

**2. Querying Phase (Search):**
- **Query Intent Classification:** When you enter a search query (e.g., "a picture of a dog" or "a document about machine learning"), the query is first passed to a fine-tuned **MobileBERT** model. This model classifies your intent as either `IMAGE` or `TEXT`.
- **Query Embedding:** Your text query is then converted into a vector embedding using the same MobileCLIP model used for indexing.
- **Similarity Search:** Based on the classified intent, the engine performs a similarity search against the corresponding FAISS index (either text or image). It uses the `HNSWFlat` index to find the vectors in the index that are closest to your query vector.
- **Display Results:** The top 5 most similar files are retrieved, and their metadata (file name and path) are displayed in a clean, user-friendly interface powered by the **Rich** library.

---

### **TECHNOLOGY STACK**

- **Core Language:** Python
- **CLI Framework:** [Rich](https://github.com/Textualize/rich) for a beautiful and interactive command-line interface.
- **AI & Machine Learning:**
    - **[PyTorch](https://pytorch.org/):** The primary deep learning framework.
    - **[MobileCLIP](https://github.com/apple/ml-mobileclip):** For generating high-quality, multimodal (text and image) embeddings efficiently.
    - **[Hugging Face Transformers](https://huggingface.co/docs/transformers/index):** Used for the `MobileBERT` model for query intent classification.
- **Vector Search:**
    - **[FAISS](https://faiss.ai/):** (Facebook AI Similarity Search) for creating, storing, and efficiently searching through the vector embeddings. The `IndexHNSWFlat` implementation is used for its speed and ability to dynamically add items.
- **File Content Extraction:**
    - `PyPDF2` for PDFs.
    - `python-docx` for Word documents.

---

### **PROJECT STRUCTURE**

```
/
├─── encoder/           # Handles file traversal, content extraction, and embedding generation.
│    ├─── main_seq.py   # Main sequential logic for the indexing pipeline.
│    ├─── embedding.py  # Generates embeddings using MobileCLIP.
│    └─── faiss_base.py # Manages the FAISS vector indexes.
│
├─── query/             # Handles the search logic.
│    ├─── query.py      # Core search logic and result presentation.
│    └─── utils.py      # Contains the query intent classification logic.
│
├─── weights/           # Stores pre-trained model weights for MobileCLIP and MobileBERT.
├─── index/             # Stores the generated FAISS indexes and metadata.
├─── run.py             # Main entry point of the application.
└─── requirements.txt   # Project dependencies.
```

### **FUTURE SCOPE**

1.  **Expanded File Support:** Add support for more file types, including videos and source code files.
2.  **Offline LLM Integration:** Integrate a small, local Large Language Model (LLM) to enable more conversational search and summarization, turning it into an "offline Perplexity."
3.  **Improved CLI/UI:** Further enhance the user interface with more advanced features and visualizations.
4.  **Performance Optimization:** Explore further parallelization of the indexing process to make it even faster on multi-core systems.
