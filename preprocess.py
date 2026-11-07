import cv2
import numpy as np
from PIL import Image
import easyocr

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks
def select_relevant_chunks(query, chunks, top_k=3):
    scored_chunks = []

    for chunk in chunks:
        score = sum(word.lower() in chunk.lower() for word in query.split())
        scored_chunks.append((score, chunk))

    scored_chunks.sort(reverse=True)

    selected = [chunk for score, chunk in scored_chunks[:top_k]]

    return selected


def preprocess_image(image):
    img = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize for better OCR accuracy
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5)

    # Remove noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh


def extract_text_from_image(image):
    reader = easyocr.Reader(['en'])
    processed = preprocess_image(image)
    result = reader.readtext(processed, detail=0)
    text = " ".join(result)
    return text
