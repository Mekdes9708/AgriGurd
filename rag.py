"""Auditable retrieval: page citations, disease filters, and zero generated agronomy claims."""
from functools import lru_cache
import re
from pathlib import Path
import fitz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
PDF = ROOT / 'knowledge' / 'ethiopia_maize_review_2023.pdf'
TITLE = 'Gidi (2023), Major Maize Diseases in Ethiopia and Their Management'
SOURCE_ID = 'GIDI2023'
DISEASE_TERMS = {
    'healthy': [],
    'northern_leaf_blight': ['turcicum leaf blight', 'northern leaf blight', 'tlb', 'exserohilum turcicum'],
    'gray_leaf_spot': ['gray leaf spot', 'grey leaf spot', 'cercospora'],
    'common_rust': ['common rust', 'maize rust', 'puccinia sorghi'],
    'maize_lethal_necrosis': ['maize lethal necrosis', 'mlnd', 'mln', 'chlorotic mottle']
}
# Page-specific evidence areas verified by the supplied review's section boundaries.
PAGE_HINTS = {
    'common_rust': [2, 3],
    'maize_lethal_necrosis': [3, 4],
    'gray_leaf_spot': [4, 5],
    'northern_leaf_blight': [5, 6],
}

def clean(text):
    return re.sub(r'\s+', ' ', text).strip()

@lru_cache(maxsize=2)
def load_passages(path=None):
    target = Path(path) if path else PDF
    chunks = []
    doc = fitz.open(target)
    for page_index, page in enumerate(doc):
        # Pages 7-10 are bibliography rather than guidance: retain only narrative sections.
        if page_index >= 6:
            continue
        blocks = page.get_text('blocks')
        for block_index, block in enumerate(blocks):
            raw = str(block[4])
            passage = clean(raw)
            if len(passage) < 95:
                continue
            # Avoid headers and figure-only labels.
            if passage.lower().startswith('copyright'):
                continue
            chunks.append({
                'id': f'{SOURCE_ID}-p{page_index+1}-b{block_index}',
                'source': TITLE,
                'page': page_index+1,
                'text': passage,
                'url': 'https://doi.org/10.23880/oajar-16000305',
            })
    doc.close()
    return chunks


def _priority(candidate, condition):
    if condition == 'healthy':
        return False
    terms = DISEASE_TERMS.get(condition, [])
    page_ok = candidate['page'] in PAGE_HINTS.get(condition, [])
    text = candidate['text'].casefold()
    return page_ok or any(term in text for term in terms)


def retrieve(query, condition, k=4, model=None, passages=None):
    """Retrieve evidence; never present text from another disease as management for this one.

    model is optional an initialized sentence_transformers.SentenceTransformer.
    """
    data = list(passages) if passages is not None else load_passages()
    if condition == 'healthy':
        return []
    pool = [p for p in data if _priority(p, condition)]
    if not pool:
        return []
    texts = [p['text'] for p in pool]
    augmented_query = ' '.join(DISEASE_TERMS.get(condition, [])[:2]) + ' ' + query
    if model is not None:
        embs = model.encode([augmented_query] + texts, normalize_embeddings=True, show_progress_bar=False)
        scores = cosine_similarity(embs[:1], embs[1:])[0]
        method = 'pretrained multilingual embeddings'
    else:
        vectors = TfidfVectorizer(stop_words='english', ngram_range=(1,2), sublinear_tf=True).fit_transform([augmented_query]+texts)
        scores = cosine_similarity(vectors[:1], vectors[1:])[0]
        method = 'TF-IDF baseline'
    order = np.argsort(scores)[::-1]
    return [{**pool[i], 'score': float(scores[i]), 'method': method} for i in order[:k] if scores[i] > 0.005]


def coverage_flag(results):
    if not results:
        return 'No relevant passage retrieved. Do not invent a recommendation.'
    return 'Research excerpts retrieved for review, not a verified field diagnosis.'
