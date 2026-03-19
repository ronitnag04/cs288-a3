# CS288 Assignment 3

This repo runs a simple retrieval-augmented QA (RAG) pipeline:
embed the cleaned corpus, retrieve top-k chunks for each question, then ask an LLM to answer using only the retrieved context.

## Expected paths
- Raw/crawled HTML: `html/*.html`
  - Optional if cleaned_text is availible
- Cleaned corpus used for embeddings: `html/cleaned_text/*.txt`
  - Download and unzip from [shared gdrive folder](https://drive.google.com/drive/folders/1s-L0gxE-MGne73tlFe1MraiguHOHJkJi?usp=drive_link)
- Embedding cache (autograder/submission artifact): `embeddings_cache/`
  - FAISS index: `embeddings_cache/<index_name>.faiss`
  - Metadata/chunk text: `embeddings_cache/<index_name>.pkl`
  - Default index name in code: `embeddings_only_index` (so `embeddings_cache/embeddings_only_index.faiss` and `.pkl`)
  - Download and unzip from [shared gdrive folder](https://drive.google.com/drive/folders/1s-L0gxE-MGne73tlFe1MraiguHOHJkJi?usp=drive_link)

## File overview
- `main.py`: loads the index, retrieves top-k chunks per question, calls the LLM, writes predictions to a file.
- `embeddings.py`: builds/loads the FAISS index from `html/cleaned_text/*.txt` and stores it in `embeddings_cache/`.
  - You may need to set HF_TOKEN to use some embedding models (like google/embeddinggemma-300m)
- `llm.py`: OpenRouter chat completion wrapper (uses `OPENROUTER_API_KEY`) used in submission environment.
- `llm_local.py`: local/dev LLM backend using Amazon Bedrock (uses `boto3`; not required by autograder).
    - Configure AWS profile with AWS IAM Access Key
- `evaluate.py`: evaluates predictions vs reference answers (average F1/precision/recall).
- `run.sh`: convenience wrapper that runs `main.py` and ensures the embedding model is available.
- `zip_submission.sh`: creates `submission.zip` (includes code + `embeddings_cache/*`).
- `crawler.ipynb`: crawls pages and writes cleaned text into `html/cleaned_text/*.txt`.
- `test_rag.ipynb`: small notebook to test chunking/retrieval with the cached index.
- `requirements.txt`: Python dependencies to create submission conda environment.

## Data folders
- `questions/`: question files (JSON-lines).
- `predictions/`: prediction files (generated outputs and provided examples).

## Usage

### 1. Setup your Environment
Create the conda environment (Python 3.10.12):
```bash
conda create -n cs288-a3 -y python=3.10.12
conda activate cs288-a3
pip install -r requirements.txt
```

If you are using a HuggingFace embedding model that require authentication, [create a User Access Token](https://huggingface.co/docs/hub/en/security-tokens) and set it in your environment. 
```bash
export HF_TOKEN=<token>
```

If you are using AWS Bedrock for local development, make sure to configure your environment. 
```bash
pip install awscli
```
```bash
aws configure
```

### 3. Data Collection
If scraping and cleaning is already complete, you can download the files from the [shared gdrive folder](https://drive.google.com/drive/folders/1s-L0gxE-MGne73tlFe1MraiguHOHJkJi?usp=drive_link). Unzip the cleaned_text.zip and make sure the `.txt` files are located at `html/cleaned_text/*.txt`.

Otherwise, follow the `crawler.ipynb` notebook to scrape and clean the HTML files.

### 4. Building the Embedding Index
If the embeddings index has already been built, you can download the files from the [shared gdrive folder](https://drive.google.com/drive/folders/1s-L0gxE-MGne73tlFe1MraiguHOHJkJi?usp=drive_link). Unzip the embeddings_cache.zip and make sure the `embeddings_only_index.faiss` and `embeddings_only_index.pkl` files are located at `embeddings_cache/`.

To build the embedding index of the scraped & cleaned files, run the `embeddings.py` script with the `build` option.
```bash
python embeddings.py build
```

This will save the embedding index for the files in `embeddings_cache/embeddings_only_index.pkl` and `embeddings_cache/embeddings_only_index.faiss`.

### 5. Use the RAG pipeline
You can now run the RAG pipeline on questions. To get predictions on a set of questions, you can use the `run.sh` script:
```bash
bash run.sh questions/test_questions.txt predictions/test_predictions.txt
```
The questions files are located at `questions/reference_questions.txt` and `questions/test_questions.txt`. You can set the predictions path to whatever you want, but we recommend a file name in the `predictions` folder.

### 6. Evaluate Results
Use the `evaluate.py` script to compare the predictions to the expected answers for the questions.
```bash
python evaluate.py -q questions/test_questions.txt -p predictions/test_predictions.txt
```
This will output the average F1, Precision, and Recall scores for the questions and predictions.
