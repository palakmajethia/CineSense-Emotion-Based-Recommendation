import argparse
import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

USE_GPU = os.getenv("CINESENSE_USE_GPU", "0") == "1"

try:
    if USE_GPU:
        import cudf  # type: ignore
        import cupy as cp  # type: ignore
        from cuml.feature_extraction.text import TfidfVectorizer as CuTfidfVectorizer  # type: ignore
        from cuml.linear_model import LogisticRegression as CuLogisticRegression  # type: ignore
        from cuml.metrics import confusion_matrix as cu_confusion_matrix  # type: ignore
        from cuml.neighbors import NearestNeighbors as CuNearestNeighbors  # type: ignore
        from cuml.metrics import pairwise_distances  # type: ignore
        GPU_AVAILABLE = True
    else:
        GPU_AVAILABLE = False
except Exception:
    GPU_AVAILABLE = False

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


DEFAULT_EMOTION_KEYWORDS: Dict[str, str] = {
    "happy": "comedy fun family uplifting joyful friendship feel-good",
    "sad": "drama emotional heartbreak grief meaningful introspective",
    "anger": "revenge conflict intense rage gritty",
    "fear": "horror suspense thriller tension survival dark",
    "excited": "action adventure fast-paced thriller high-stakes",
    "calm": "slow peaceful soothing romance warm slice-of-life",
    "surprise": "twist mystery unpredictable mind-bending",
    "disgust": "dark satire disturbing crime psychological",
}


@dataclass
class EmotionModel:
    vectorizer: object
    model: object
    is_gpu: bool


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [tok for tok in text.split() if tok not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def load_emotion_data(path: Optional[Path]) -> pd.DataFrame:
    if path and path.exists():
        df = pd.read_csv(path)
    else:
        fallback = Path("data/emotion_dataset.csv")
        if not fallback.exists():
            raise FileNotFoundError("No emotion dataset found. Provide --emotion-data or create data/emotion_dataset.csv")
        df = pd.read_csv(fallback)

    required = {"text", "emotion"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Emotion dataset missing columns: {missing}")

    df = df.dropna(subset=["text", "emotion"]).copy()
    df["clean_text"] = df["text"].map(clean_text)
    return df[df["clean_text"].str.len() > 0]


def train_emotion_classifier(emotion_df: pd.DataFrame, random_state: int = 42) -> Tuple[EmotionModel, Dict[str, object]]:
    x_train, x_test, y_train, y_test = train_test_split(
        emotion_df["clean_text"],
        emotion_df["emotion"],
        test_size=0.25,
        random_state=random_state,
        stratify=emotion_df["emotion"],
    )

    if GPU_AVAILABLE:
        vectorizer = CuTfidfVectorizer(ngram_range=(1, 2), min_df=1)
        x_train_vec = vectorizer.fit_transform(cudf.Series(x_train.tolist()))
        x_test_vec = vectorizer.transform(cudf.Series(x_test.tolist()))

        model = CuLogisticRegression(max_iter=2000)
        model.fit(x_train_vec, cudf.Series(y_train.tolist()))

        y_pred = model.predict(x_test_vec).to_pandas()
        y_test_cpu = y_test.reset_index(drop=True)
        metrics = {
            "accuracy": float(accuracy_score(y_test_cpu, y_pred)),
            "confusion_matrix": cu_confusion_matrix(y_test_cpu, y_pred).to_pandas().values.tolist(),
            "labels": sorted(y_test_cpu.unique().tolist()),
            "classification_report": classification_report(y_test_cpu, y_pred, output_dict=True),
        }

        return EmotionModel(vectorizer=vectorizer, model=model, is_gpu=True), metrics

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(x_train_vec, y_train)

    y_pred = model.predict(x_test_vec)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "labels": sorted(y_test.unique().tolist()),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    return EmotionModel(vectorizer=vectorizer, model=model, is_gpu=False), metrics


def save_emotion_model(emotion_model: EmotionModel, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": emotion_model.vectorizer,
            "model": emotion_model.model,
            "is_gpu": emotion_model.is_gpu,
        },
        model_path,
    )


def load_emotion_model(model_path: Path) -> EmotionModel:
    payload = joblib.load(model_path)
    return EmotionModel(
        vectorizer=payload["vectorizer"],
        model=payload["model"],
        is_gpu=payload.get("is_gpu", False),
    )


def predict_emotion(text: str, emotion_model: EmotionModel) -> str:
    clean = clean_text(text)
    if emotion_model.is_gpu:
        vec = emotion_model.vectorizer.transform(cudf.Series([clean]))
        return str(emotion_model.model.predict(vec).to_pandas()[0])

    vec = emotion_model.vectorizer.transform([clean])
    return str(emotion_model.model.predict(vec)[0])


def parse_cell_list(cell: object) -> str:
    if pd.isna(cell):
        return ""
    if isinstance(cell, list):
        values = []
        for item in cell:
            if isinstance(item, dict):
                values.append(str(item.get("name", "")))
            else:
                values.append(str(item))
        return " ".join(v for v in values if v)

    text = str(cell)
    text = text.strip()
    if not text:
        return ""

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                values = []
                for item in parsed:
                    if isinstance(item, dict):
                        values.append(str(item.get("name", "")))
                    else:
                        values.append(str(item))
                return " ".join(v for v in values if v)
        except Exception:
            continue

    return text


def load_movies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"title", "overview", "genres", "keywords"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Movie dataset missing columns: {missing}")

    if "id" not in df.columns:
        df["id"] = np.arange(len(df))

    df = df.dropna(subset=["title", "overview"]).copy()
    df["genres_text"] = df["genres"].map(parse_cell_list)
    df["keywords_text"] = df["keywords"].map(parse_cell_list)

    df["combined_text"] = (
        df["genres_text"].fillna("")
        + " "
        + df["overview"].fillna("")
        + " "
        + df["keywords_text"].fillna("")
    ).map(clean_text)

    return df[df["combined_text"].str.len() > 0].reset_index(drop=True)


def build_content_matrix(movies_df: pd.DataFrame):
    if GPU_AVAILABLE:
        vectorizer = CuTfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(cudf.Series(movies_df["combined_text"].tolist()))
        return vectorizer, matrix

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(movies_df["combined_text"])
    return vectorizer, matrix


def emotion_content_scores(
    emotion: str,
    content_vectorizer: object,
    movie_matrix,
    emotion_keywords: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    mapping = emotion_keywords or DEFAULT_EMOTION_KEYWORDS
    query = mapping.get(emotion.lower(), emotion)

    if GPU_AVAILABLE:
        query_vec = content_vectorizer.transform(cudf.Series([clean_text(query)]))
        sims = 1 - pairwise_distances(query_vec, movie_matrix, metric="cosine")
        return cp.asnumpy(sims).flatten()

    query_vec = content_vectorizer.transform([clean_text(query)])
    sims = cosine_similarity(query_vec, movie_matrix).flatten()
    return sims


def normalize_title(title: str) -> str:
    if not isinstance(title, str):
        return ""
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", title)
    title = re.sub(r"[^a-z0-9\\s]", " ", title.lower())
    return " ".join(title.split())


def map_movielens_to_tmdb(
    ratings_df: pd.DataFrame,
    movielens_movies_path: Path,
    movies_df: pd.DataFrame,
) -> pd.DataFrame:
    if not movielens_movies_path.exists():
        return ratings_df

    ml_movies = pd.read_csv(movielens_movies_path)
    if "movieId" not in ml_movies.columns or "title" not in ml_movies.columns:
        return ratings_df

    ml_movies["norm_title"] = ml_movies["title"].map(normalize_title)
    tmdb_titles = movies_df[["id", "title"]].copy()
    tmdb_titles["norm_title"] = tmdb_titles["title"].map(normalize_title)

    tmdb_map = dict(zip(tmdb_titles["norm_title"], tmdb_titles["id"]))
    ml_movies["tmdb_id"] = ml_movies["norm_title"].map(tmdb_map)

    mapping = ml_movies.dropna(subset=["tmdb_id"])[["movieId", "tmdb_id"]]
    if mapping.empty:
        return ratings_df

    ratings_df = ratings_df.merge(mapping, on="movieId", how="inner")
    ratings_df = ratings_df.drop(columns=["movieId"]).rename(columns={"tmdb_id": "movieId"})
    ratings_df = ratings_df[["userId", "movieId", "rating"]]
    return ratings_df


def load_ratings(path: Optional[Path], movies_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    if path and path.exists():
        ratings = pd.read_csv(path)
        required = {"userId", "movieId", "rating"}
        missing = required.difference(ratings.columns)
        if missing:
            raise ValueError(f"Ratings dataset missing columns: {missing}")
        movielens_movies = path.parent / "movies.csv"
        ratings = map_movielens_to_tmdb(ratings, movielens_movies, movies_df)
        return ratings

    rng = np.random.default_rng(seed=random_state)
    movie_ids = movies_df["id"].tolist()
    if len(movie_ids) < 20:
        raise ValueError("Need at least 20 movies to synthesize ratings")

    rows = []
    user_count = 100
    for user_id in range(1, user_count + 1):
        picks = rng.choice(movie_ids, size=20, replace=False)
        for movie_id in picks:
            rating = float(np.clip(rng.normal(3.5, 1.0), 0.5, 5.0))
            rows.append((user_id, int(movie_id), round(rating * 2) / 2))

    return pd.DataFrame(rows, columns=["userId", "movieId", "rating"])


def build_user_item_matrix(ratings_df: pd.DataFrame):
    user_ids = ratings_df["userId"].unique()
    movie_ids = ratings_df["movieId"].unique()

    user_index = {uid: i for i, uid in enumerate(user_ids)}
    movie_index = {mid: i for i, mid in enumerate(movie_ids)}

    row_idx = ratings_df["userId"].map(user_index).to_numpy()
    col_idx = ratings_df["movieId"].map(movie_index).to_numpy()
    data = ratings_df["rating"].to_numpy(dtype=float)

    matrix = sp.csr_matrix(
        (data, (row_idx, col_idx)),
        shape=(len(user_ids), len(movie_ids)),
    )

    return matrix, user_ids, movie_ids


def fit_knn(user_item_matrix, n_neighbors: int = 5):
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    model.fit(user_item_matrix)
    return model


def collaborative_scores_for_user(
    user_id: int,
    ratings_df: pd.DataFrame,
    user_item_matrix,
    user_ids: np.ndarray,
    knn,
) -> pd.Series:
    if user_id not in set(user_ids):
        return pd.Series(dtype=float)

    user_pos = int(np.where(user_ids == user_id)[0][0])
    user_vector = user_item_matrix[user_pos]

    distances, indices = knn.kneighbors(user_vector)
    distances = distances.flatten()
    indices = indices.flatten()

    neighbors = user_ids[indices].tolist()
    weights = 1 - distances

    neighbor_ratings = ratings_df[ratings_df["userId"].isin(neighbors)].copy()
    neighbor_ratings["weight"] = neighbor_ratings["userId"].map(dict(zip(neighbors, weights)))
    neighbor_ratings["weighted_rating"] = neighbor_ratings["rating"] * neighbor_ratings["weight"]

    score = neighbor_ratings.groupby("movieId")["weighted_rating"].sum() / (
        neighbor_ratings.groupby("movieId")["weight"].sum() + 1e-9
    )
    return score


def recommend_movies(
    user_text: str,
    movies_df: pd.DataFrame,
    emotion_model: EmotionModel,
    ratings_df: pd.DataFrame,
    top_n_content: int = 10,
    final_n: int = 5,
    user_id: Optional[int] = None,
    alpha_content: float = 0.7,
) -> Tuple[str, pd.DataFrame]:
    emotion = predict_emotion(user_text, emotion_model)

    content_vectorizer, movie_matrix = build_content_matrix(movies_df)
    c_scores = emotion_content_scores(emotion, content_vectorizer, movie_matrix)

    movies = movies_df.copy()
    movies["content_score"] = c_scores

    user_item, user_ids, _ = build_user_item_matrix(ratings_df)
    knn = fit_knn(user_item, n_neighbors=min(5, len(user_ids)))

    if user_id is not None and user_id in set(user_ids):
        collab = collaborative_scores_for_user(user_id, ratings_df, user_item, user_ids, knn)
        movies = movies.merge(collab.rename("collab_score"), left_on="id", right_index=True, how="left")
        movies["collab_score"] = movies["collab_score"].fillna(movies["collab_score"].mean())
        movies["final_score"] = alpha_content * movies["content_score"] + (1 - alpha_content) * movies["collab_score"]
    else:
        movies["collab_score"] = np.nan
        movies["final_score"] = movies["content_score"]

    top_content = movies.nlargest(top_n_content, "content_score")
    final = top_content.nlargest(final_n, "final_score").copy()

    max_content = max(final["content_score"].max(), 1e-9)
    final["similarity_pct"] = (final["content_score"] / max_content * 100).round(1)
    final["explanation"] = final["similarity_pct"].map(
        lambda p: f'Recommended because it matches your "{emotion}" emotion and shares {p}% similarity with {emotion} keywords.'
    )

    return emotion, final[["title", "final_score", "content_score", "collab_score", "explanation"]]


def content_recommend_for_user(
    user_id: int,
    ratings_train: pd.DataFrame,
    movies_df: pd.DataFrame,
    content_vectorizer,
    movie_matrix,
    top_k: int = 10,
) -> List[int]:
    user_hist = ratings_train[(ratings_train["userId"] == user_id) & (ratings_train["rating"] >= 4.0)]
    if user_hist.empty:
        return []

    id_to_index = {mid: i for i, mid in enumerate(movies_df["id"].tolist())}
    profile_vec = None
    for _, row in user_hist.iterrows():
        mid = row["movieId"]
        if mid not in id_to_index:
            continue
        vec = movie_matrix[id_to_index[mid]]
        profile_vec = vec if profile_vec is None else profile_vec + vec

    if profile_vec is None:
        return []

    if GPU_AVAILABLE:
        sims = 1 - pairwise_distances(profile_vec, movie_matrix, metric="cosine")
        sims = cp.asnumpy(sims).flatten()
    else:
        sims = cosine_similarity(profile_vec, movie_matrix).flatten()

    watched = set(ratings_train[ratings_train["userId"] == user_id]["movieId"].tolist())

    rec_ids = []
    ranked_idx = np.argsort(-sims)
    for idx in ranked_idx:
        mid = int(movies_df.iloc[idx]["id"])
        if mid not in watched:
            rec_ids.append(mid)
        if len(rec_ids) >= top_k:
            break
    return rec_ids


def hybrid_recommend_for_user(
    user_id: int,
    ratings_train: pd.DataFrame,
    movies_df: pd.DataFrame,
    content_vectorizer,
    movie_matrix,
    top_k: int = 10,
    alpha: float = 0.6,
) -> List[int]:
    content_ids = content_recommend_for_user(
        user_id=user_id,
        ratings_train=ratings_train,
        movies_df=movies_df,
        content_vectorizer=content_vectorizer,
        movie_matrix=movie_matrix,
        top_k=top_k * 3,
    )

    if not content_ids:
        return []

    user_item, user_ids, _ = build_user_item_matrix(ratings_train)
    if user_id not in set(user_ids) or len(user_ids) < 2:
        return content_ids[:top_k]

    knn = fit_knn(user_item, n_neighbors=min(5, len(user_ids)))
    collab_scores = collaborative_scores_for_user(user_id, ratings_train, user_item, user_ids, knn)

    id_to_idx = {mid: i for i, mid in enumerate(movies_df["id"].tolist())}
    content_score_map = {}

    user_hist = ratings_train[(ratings_train["userId"] == user_id) & (ratings_train["rating"] >= 4.0)]
    profile_vec = None
    for _, row in user_hist.iterrows():
        mid = row["movieId"]
        if mid not in id_to_idx:
            continue
        vec = movie_matrix[id_to_idx[mid]]
        profile_vec = vec if profile_vec is None else profile_vec + vec

    if profile_vec is not None:
        if GPU_AVAILABLE:
            sims = 1 - pairwise_distances(profile_vec, movie_matrix, metric="cosine")
            sims = cp.asnumpy(sims).flatten()
        else:
            sims = cosine_similarity(profile_vec, movie_matrix).flatten()

        for mid in content_ids:
            if mid in id_to_idx:
                content_score_map[mid] = sims[id_to_idx[mid]]

    rows = []
    for mid in content_ids:
        c = content_score_map.get(mid, 0.0)
        k = float(collab_scores.get(mid, 0.0))
        final = alpha * c + (1 - alpha) * k
        rows.append((mid, final))

    rows.sort(key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in rows[:top_k]]


def precision_recall_at_k(
    recs: Dict[int, List[int]],
    relevant: Dict[int, List[int]],
    k: int,
) -> Tuple[float, float]:
    users = [u for u in recs if u in relevant and relevant[u]]
    if not users:
        return 0.0, 0.0

    precision_vals = []
    recall_vals = []
    for u in users:
        recommended = recs[u][:k]
        rel_set = set(relevant[u])
        hits = len(set(recommended).intersection(rel_set))
        precision_vals.append(hits / k)
        recall_vals.append(hits / max(1, len(rel_set)))

    return float(np.mean(precision_vals)), float(np.mean(recall_vals))


def evaluate_recommenders(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    k: int = 5,
    min_user_ratings: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(random_state)

    eligible_users = []
    held_out = {}
    train_parts = []

    for uid, grp in ratings_df.groupby("userId"):
        if len(grp) < min_user_ratings:
            train_parts.append(grp)
            continue

        liked = grp[grp["rating"] >= 4.0]
        if liked.empty:
            train_parts.append(grp)
            continue

        eligible_users.append(uid)
        hold_row = liked.sample(n=1, random_state=int(rng.integers(1, 1_000_000))).iloc[0]
        held_out[int(uid)] = [int(hold_row["movieId"])]
        remaining = grp[grp.index != hold_row.name]
        train_parts.append(remaining)

    ratings_train = pd.concat(train_parts, ignore_index=True)

    content_vectorizer, movie_matrix = build_content_matrix(movies_df)

    content_recs = {}
    hybrid_recs = {}

    for uid in eligible_users:
        content_recs[int(uid)] = content_recommend_for_user(
            user_id=int(uid),
            ratings_train=ratings_train,
            movies_df=movies_df,
            content_vectorizer=content_vectorizer,
            movie_matrix=movie_matrix,
            top_k=k,
        )
        hybrid_recs[int(uid)] = hybrid_recommend_for_user(
            user_id=int(uid),
            ratings_train=ratings_train,
            movies_df=movies_df,
            content_vectorizer=content_vectorizer,
            movie_matrix=movie_matrix,
            top_k=k,
        )

    p_content, r_content = precision_recall_at_k(content_recs, held_out, k)
    p_hybrid, r_hybrid = precision_recall_at_k(hybrid_recs, held_out, k)

    return {
        "precision_at_k_content": p_content,
        "recall_at_k_content": r_content,
        "precision_at_k_hybrid": p_hybrid,
        "recall_at_k_hybrid": r_hybrid,
        "users_evaluated": float(len(eligible_users)),
    }


def pretty_confusion(labels: List[str], matrix: List[List[int]]) -> str:
    label_row = "\t".join(["true\\pred"] + labels)
    rows = [label_row]
    for i, row in enumerate(matrix):
        rows.append("\t".join([labels[i]] + [str(v) for v in row]))
    return "\n".join(rows)


def run_pipeline(args: argparse.Namespace) -> None:
    movies_df = load_movies(Path(args.movies_data))

    emotion_df = load_emotion_data(Path(args.emotion_data) if args.emotion_data else None)
    emotion_model, emotion_metrics = train_emotion_classifier(emotion_df)
    save_emotion_model(emotion_model, Path(args.model_out))

    ratings_df = load_ratings(Path(args.ratings_data) if args.ratings_data else None, movies_df)

    predicted_emotion, recs = recommend_movies(
        user_text=args.text,
        movies_df=movies_df,
        emotion_model=emotion_model,
        ratings_df=ratings_df,
        top_n_content=args.top_n_content,
        final_n=args.final_n,
        user_id=args.user_id,
        alpha_content=args.alpha,
    )

    eval_scores = evaluate_recommenders(movies_df, ratings_df, k=args.eval_k)

    print("\n=== Emotion Classifier (Logistic Regression) ===")
    if GPU_AVAILABLE:
        print("Mode: GPU (cuML)")
    else:
        print("Mode: CPU (scikit-learn)")
    print(f"Accuracy: {emotion_metrics['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(pretty_confusion(emotion_metrics["labels"], emotion_metrics["confusion_matrix"]))

    print("\n=== User Input ===")
    print(args.text)
    print(f"Predicted Emotion: {predicted_emotion}")

    print("\n=== Final Top Recommendations ===")
    for i, (_, row) in enumerate(recs.iterrows(), start=1):
        print(f"{i}. {row['title']}")
        print(f"   final_score={row['final_score']:.4f}, content_score={row['content_score']:.4f}")
        if pd.notna(row["collab_score"]):
            print(f"   collab_score={row['collab_score']:.4f}")
        print(f"   {row['explanation']}")

    print("\n=== Evaluation (Precision@K / Recall@K) ===")
    print(f"Content  -> Precision@{args.eval_k}: {eval_scores['precision_at_k_content']:.4f}, Recall@{args.eval_k}: {eval_scores['recall_at_k_content']:.4f}")
    print(f"Hybrid   -> Precision@{args.eval_k}: {eval_scores['precision_at_k_hybrid']:.4f}, Recall@{args.eval_k}: {eval_scores['recall_at_k_hybrid']:.4f}")
    print(f"Users evaluated: {int(eval_scores['users_evaluated'])}")
    print(f"Saved emotion model to: {args.model_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emotion-based Hybrid Movie Recommender")
    parser.add_argument("--movies-data", required=True, help="Path to movies_metadata.csv with columns: title, overview, genres, keywords")
    parser.add_argument("--ratings-data", default=None, help="Optional path to ratings.csv with columns: userId, movieId, rating")
    parser.add_argument("--emotion-data", default=None, help="Optional path to emotion dataset csv with columns: text, emotion")
    parser.add_argument("--model-out", default="models/emotion_lr.joblib", help="Output path for saved logistic regression model")
    parser.add_argument("--text", required=True, help="User input text for emotion detection")
    parser.add_argument("--user-id", type=int, default=None, help="Optional userId for personalized hybrid reranking")
    parser.add_argument("--top-n-content", type=int, default=10, help="Top-N content-based candidates")
    parser.add_argument("--final-n", type=int, default=5, help="Final recommendations to return")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for content score in hybrid (0..1)")
    parser.add_argument("--eval-k", type=int, default=5, help="K for Precision@K and Recall@K")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
