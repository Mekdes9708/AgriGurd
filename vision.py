"""Maize photo model inference; pixel normalization lives INSIDE the saved Keras model."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

BASE = Path(__file__).parent
DEFAULT_LABELS = ['healthy','northern_leaf_blight','gray_leaf_spot','common_rust']


def load_labels(path=None):
    target = Path(path) if path else BASE/'models'/'labels.json'
    if target.exists():
        labels = json.loads(target.read_text(encoding='utf-8'))
        if not isinstance(labels, list) or len(labels)!=4 or set(labels)!=set(DEFAULT_LABELS):
            raise ValueError('Model labels do not match four expected categories')
        return labels
    return DEFAULT_LABELS


def preprocess(image):
    im = ImageOps.exif_transpose(image).convert('RGB').resize((224,224))
    return np.expand_dims(np.asarray(im, dtype=np.float32),0)


def predict(model, image, labels=None):
    labels = labels or DEFAULT_LABELS
    raw = np.asarray(model.predict(preprocess(image), verbose=0))[0]
    if raw.shape!=(len(labels),) or not np.all(np.isfinite(raw)):
        raise ValueError('Unexpected model output')
    if np.any(raw < 0) or not np.isclose(raw.sum(), 1.0, atol=0.03):
        raise ValueError('Output does not look like four softmax scores')
    order = np.argsort(raw)[::-1]
    top = float(raw[order[0]])
    margin = top - float(raw[order[1]])
    uncertain = top < 0.70 or margin < 0.20
    return dict(label=labels[int(order[0])], top_score=top, margin=margin,
                uncertain=uncertain, ranks=[(labels[int(i)], float(raw[i])) for i in order])
