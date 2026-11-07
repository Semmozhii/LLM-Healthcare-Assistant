from Bio import Entrez
import time

Entrez.email = "semmozhisakthi@example.com"

def get_medical_evidence(query, max_results=3):
    try:
        # Add delay to avoid rate limit
        time.sleep(1)

        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results
        )

        record = Entrez.read(handle)
        handle.close()

        pmids = record["IdList"]

        if not pmids:
            return None, []

        # Fetch abstracts
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(pmids),
            rettype="abstract",
            retmode="text"
        )

        abstracts = handle.read()
        handle.close()

        return abstracts[:3000], pmids

    except Exception as e:
        print("PubMed API Error:", e)
        return None, []
