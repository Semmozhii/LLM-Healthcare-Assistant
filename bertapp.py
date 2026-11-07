''''from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load BioBERT
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def embed_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.numpy()

def select_relevant_abstracts(query, abstracts, top_k=2):
    sentences = abstracts.split(". ")

    query_embedding = embed_text(query)

    sentence_embeddings = [embed_text(s) for s in sentences]

    scores = []

    for emb in sentence_embeddings:
        score = cosine_similarity(query_embedding, emb)[0][0]
        scores.append(score)

    ranked = sorted(
        zip(sentences, scores),
        key=lambda x: x[1],
        reverse=True
    )

    best_sentences = [r[0] for r in ranked[:top_k]]

    return " ".join(best_sentences)'''