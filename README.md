# Semantic Book Recommender

A book recommendation app that finds books from a natural-language description (e.g. *"A story about forgiveness"*) instead of keywords. Results can be filtered by category and re-ranked by emotional tone, and are shown as a gallery of covers with short captions.

The project combines three techniques:

- **Semantic search:** OpenAI embeddings stored in a Chroma vector database.
- **Zero-shot text classification:** `facebook/bart-large-mnli` assigns missing categories.
- **Emotion analysis:** `j-hartmann/emotion-english-distilroberta-base` scores each description for anger, disgust, fear, joy, neutral, sadness and surprise.

The interface is built with [Gradio](https://www.gradio.app/).

## How It Works

1. **Retrieve:** the user's query is embedded and the 50 most similar book descriptions are fetched from Chroma.
2. **Filter:** results are optionally restricted to a category (Fiction, Nonfiction, Children's Fiction, Children's Nonfiction).
3. **Re-rank:** results are optionally sorted by an emotional tone, and the top 16 are shown.

| Tone in UI  | Emotion score used |
|-------------|--------------------|
| Happy       | `joy`              |
| Surprising  | `surprise`         |
| Angry       | `anger`            |
| Suspenseful | `fear`             |
| Sad         | `sadness`          |

## Project Structure

| File | Purpose |
|------|---------|
| `data-exploration.ipynb` | Downloads the dataset, explores missing values, and cleans the data. Outputs `books_cleaned.csv`. |
| `semantic-book-recommender.ipynb` | Builds the vector database and tests semantic search. Maps categories and classifies the rest with zero-shot classification. Outputs `books_with_categories.csv` and `tagged_description.txt`. |
| `sentiment-analysis.ipynb` | Computes per-emotion scores for every book. Outputs `books_with_emotions.csv`. |
| `gradio-dashboard.py` | The Gradio web app. |
| `books_with_emotions.csv` | Final dataset used by the app. |
| `tagged_description.txt` | One `ISBN13 + description` line per book, used to build the vector store. |
| `chroma_db/` | Persisted Chroma vector store (created on first run). |
| `cover-not-found.jpg` | Fallback image for books without a thumbnail. |
| `.env` | Holds your `OPENAI_API_KEY` (not committed). |

## Data Pipeline

### 1. Data exploration and cleaning
- Source: the Kaggle dataset [`dylanjcastillo/7k-books-with-metadata`](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) (about 6,800 books).
- Rows missing a description, page count, rating or publication year are dropped. A Spearman correlation check showed missing descriptions were not strongly related to the other numeric features.
- Descriptions shorter than 25 words are removed, since shorter ones were mostly uninformative (e.g. "Donation." or "No Marketing Blurb"). About 5,200 books remain.
- `title` and `subtitle` are merged into `title_and_subtitle`.
- A `tagged_description` column (`ISBN13` followed by the description) is created so that each retrieved chunk can be mapped back to its book.

### 2. Categories and vector search
- Descriptions are split one-per-line using `CharacterTextSplitter` (separator `"\n"`, `chunk_size=1`), embedded with `OpenAIEmbeddings`, and persisted in Chroma.
- The 12 most common raw categories are mapped to four simple categories. The remaining ~1,450 books are classified as **Fiction** or **Nonfiction** with zero-shot classification, which scored about **78% accuracy** on a 600-book labelled sample.

### 3. Emotion analysis
- Each description is split into sentences and every sentence is scored for the seven emotions.
- For each emotion, the **maximum** score across sentences is kept, so a description is not judged by its overall average.

## Setup

### Prerequisites
- Python 3.10+ (developed on 3.12)
- An OpenAI API key (used for embeddings)
- Optional: a Kaggle setup that works with `kagglehub`, and a GPU to speed up the Hugging Face models

### Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install pandas numpy python-dotenv gradio \
            langchain-community langchain-openai langchain-text-splitters langchain-chroma \
            transformers torch kagglehub seaborn matplotlib tqdm jupyter
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Option A: Run the app with the included data
If `books_with_emotions.csv` is already present:

```bash
python gradio-dashboard.py
```

On the first run, if `chroma_db/` doesn't exist, the app builds it from `tagged_description.txt`. This calls the OpenAI embeddings API for every book, so it takes a little time and uses API credits. Later runs reuse the saved store.

The app launches with `share=True`, so Gradio prints a local URL and a temporary public link.

### Option B: Rebuild everything from scratch
Run the notebooks in this order:

1. `data-exploration.ipynb`, which produces `books_cleaned.csv`
2. `semantic-book-recommender.ipynb`, which produces `tagged_description.txt` and `books_with_categories.csv`
3. `sentiment-analysis.ipynb`, which produces `books_with_emotions.csv`

Then start the app as in Option A.

> The zero-shot classification and emotion scoring steps run a transformer over thousands of descriptions and can take 30 minutes or more on CPU.

## Example Queries

- *A story about forgiveness*
- *A book to teach children about nature*
- *A suspenseful mystery set in a small town*

Try combining a query with a category (e.g. Children's Nonfiction) or a tone (e.g. Suspenseful).

## Tech Stack

Python, pandas, NumPy, LangChain, OpenAI Embeddings, ChromaDB, Hugging Face Transformers, Gradio, seaborn/matplotlib.

## Known Limitations

- Zero-shot Fiction/Nonfiction classification is about 78% accurate, so some categories are wrong.
- Emotion scores come from a sentence-level model and are a rough signal, not a ground truth.
- Tone sorting runs only after the top 50 semantic matches are retrieved, so it reorders relevant results rather than searching by emotion.
- The category filter is applied after retrieval, so a narrow category may return fewer than 16 books.

## Acknowledgements

- Dataset: [7k Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) by Dylan Castillo
- Models: [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli) and [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)