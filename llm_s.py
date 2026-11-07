import requests

# config
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi"

# question classify
def classify_question(query):
    """
    Classify the user medical question into a category.
    """
    prompt = f"""
Classify the following medical question into ONE category:
- Definition
- Causes
- Symptoms
- Risk Factors
- Prevention
- Treatment
- Prognosis

Question:
{query}

Return only the category name.
"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        return response.json().get("response", "Unknown").strip()

    except requests.exceptions.Timeout:
        return "Classification timed out."


#llm
def healthcare_llama(query, abstracts, pmids):
    """
    Generate a concise, evidence-based explanation using only the provided abstracts.
    The output is simplified and cites the relevant PMIDs.
    """
    prompt = f"""
You are a helpful healthcare assistant.

Rules:
- Use ONLY the provided medical evidence (do NOT hallucinate)
- Do NOT diagnose or prescribe
- Keep explanations simple and easy to understand
- Use bullet points if possible
- Cite the PMID after each key point
- Limit the answer to 3-5 sentences

Question:
{query}

Medical Evidence:
{abstracts}

Available PMIDs:
{pmids}

Provide a clear, concise, evidence-based answer:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        return response.json().get("response", "No response generated.")

    except requests.exceptions.Timeout:
        return "The model is taking too long. Please try again."
