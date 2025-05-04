# Assignment 2: Vector Database and Retrieval-Augmented Generation

## Author

**Name:** Aleksandr Jan Smoliakov  
**Email:** <aleksandr.smoliakov@mif.stud.vu.lt>  
**Student ID:** 2416123  
**Date:** 2025-05-04  

This directory contains a set of Jupyter notebooks that together implement a complete Retrieval-Augmented Generation (RAG) pipeline over Wikipedia article summaries. The workflow consists of:

1. Downloading article titles and page content from Wikipedia and saving them as Parquet.
2. Splitting article summaries into overlapping text chunks.
3. Embedding each chunk with a local MiniLM model via GPT4AllEmbeddings.
4. Indexing embeddings and metadata in a ChromaDB vector store.
5. Building a RAG pipeline that retrieves relevant passages and generates grounded answers using a local LLaMA-3 instruct model.

---

## Directory Structure

```
.
├── data/
│   ├── input/
│   │   └── wikipedia_articles.parquet      # Saved article titles, summaries, URLs
│   └── database/
│       └── wikipedia.db                    # ChromaDB persistent vector store
│
├── notebooks/
│   ├── download_content.ipynb              # Step 1: Fetch Wikipedia articles
│   ├── index_documents.ipynb               # Step 2: Chunk, embed, and index
│   └── rag_pipeline.ipynb                  # Step 3: Define & test RAG pipeline
│
├── poetry.lock                             # Poetry lock file
├── pyproject.toml                          # Poetry dependencies
└── README.md                               # This file
```

---

## Requirements

- **Python Version:** 3.10 or 3.11 (other versions may work, but were not tested)
- **Third-Party Libraries:**

  * `pandas`, `numpy`, `wikipedia`, `tqdm`
  * `chromadb`, `langchain_chroma`, `langchain_core`
  * `langchain_community` (for `GPT4AllEmbeddings` and `GPT4All` LLM)
  * `langgraph`

Dependencies are managed using [Poetry](https://python-poetry.org/) and are listed in `pyproject.toml`.

---

## How to Run

1. **Clone the repo**

    ```bash
    git clone https://github.com/spacegrapefruit/msc-studies.git
    cd msc-studies/deep_learning/assignment2
    ```

2. **Install dependencies using Poetry**

    ```bash
    poetry install -vvv
    ```

3. **Download Wikipedia articles**

   * Open **`notebooks/download_content.ipynb`**
   * Run all cells to fetch titles and page contents, producing
     `data/input/wikipedia_articles.parquet`

4. **Build the vector index**

   * Open **`notebooks/index_documents.ipynb`**
   * Run all cells to split summaries, generate embeddings, and ingest into
     `data/database/wikipedia.db`

5. **Run the RAG pipeline**

   * Open **`notebooks/rag_pipeline.ipynb`**
   * Run all cells to define the retrieval+generation graph, visualize it, and execute sample queries

---

## Outputs

* **Parquet dataset:**
  `data/input/wikipedia_articles.parquet`
  Contains article titles, summaries, full content, and URLs.

* **Vector store:**
  `data/database/wikipedia.db`
  A persistent ChromaDB collection of embeddings and metadata.

* **RAG pipeline**
  A LangGraph-constructed workflow that performs similarity search and LLM-based answer generation.
