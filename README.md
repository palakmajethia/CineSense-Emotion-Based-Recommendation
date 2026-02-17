# CineSense: Emotion-Based Hybrid Movie Recommender

This project implements your full pipeline:

1. Emotion classification with `TF-IDF + LogisticRegression`
2. Content-based recommendation with `TF-IDF + cosine similarity`
3. Collaborative filtering with `KNN (cosine)`
4. Hybrid reranking and explainable recommendations
5. Evaluation with:
   - Accuracy + Confusion Matrix (emotion model)
   - Precision@K / Recall@K (content vs hybrid)

## Files

- `cinesense_pipeline.py`: End-to-end pipeline
- `data/emotion_dataset.csv`: fallback emotion dataset (small labeled dataset)
- `requirements.txt`
- `models/`: saved trained emotion model

## Required Movie Dataset

Use your TMDB-like CSV (`movies_metadata.csv`) with columns:

- `title`
- `overview`
- `genres`
- `keywords`

Optional but recommended:

- `id` (used as `movieId`; if missing, row index is used)

`genres` and `keywords` can be plain text or TMDB-style JSON/list strings.

## Optional Ratings Dataset (for KNN)

CSV columns:

- `userId`
- `movieId`
- `rating`

If not provided, synthetic ratings are generated so the pipeline can still run.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python cinesense_pipeline.py \
  --movies-data path/to/movies_metadata.csv \
  --ratings-data path/to/ratings.csv \
  --emotion-data path/to/emotion_dataset.csv \
  --text "I feel low and want something emotional" \
  --user-id 10
```

Minimal run (uses fallback emotion data + synthetic ratings):

```bash
python cinesense_pipeline.py \
  --movies-data path/to/movies_metadata.csv \
  --text "I feel happy and want fun"
```

## Output Includes

- Predicted emotion label from user text
- Top 5 recommendations (from top 10 content candidates, reranked by KNN when user available)
- Explainability line, e.g.:

> Recommended because it matches your "sad" emotion and shares 82.0% similarity with sad keywords.

- Emotion model metrics:
  - Accuracy
  - Confusion matrix
- Recommender metrics comparison:
  - Precision@K and Recall@K for content-only vs hybrid

## Architecture Flow (Presentation)

User Input
-> Text Preprocessing
-> TF-IDF
-> Logistic Regression
-> Emotion Detected
-> Movie TF-IDF Matrix
-> Cosine Similarity
-> KNN Collaborative Filtering
-> Final Recommendation
