# AgriGuard AI v0.2 — maize vision + evidence-backed research guidance

A **research demonstration**, not a validated disease-diagnosis or treatment tool.

## What is included

- `app.py`: phone-friendly Streamlit workflow (camera/upload → top model predictions → conservative guidance → sourced research excerpts → JSON exports).
- `vision.py`: checks real `.keras` softmax outputs, preserves the 224×224 RGB, 0–255 input contract. **Do not divide by 255 again**; the trained model contains its own Rescaling preprocessing.
- `rag.py`: extracts page-anchored excerpts from the user-provided Ethiopian review and ranks relevant passages using TF-IDF; optional pretrained multilingual semantic embeddings.
- `knowledge/ethiopia_maize_review_2023.pdf`: source article provided by the project owner; Gidi (2023), DOI: 10.23880/oajar-16000305.
- `knowledge/guidance.json`: cautious, short human-review-ready guidance for common rust, gray leaf spot, northern leaf blight, and apparently healthy foliage.
- `test_agriguard.py`: logic and citation tests, no ML model required.

## Install on GitHub Codespaces

Create a new repository `AgriGuardAI` or keep these files in a separate app folder. Upload the contents of this ZIP into the project root.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -p 'test*.py' -v
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Connect the model you already trained in Colab

From your Colab export ZIP, copy these exact files:

```text
AgriGuardAI/
  app.py
  vision.py
  rag.py
  requirements.txt
  knowledge/
    ethiopia_maize_review_2023.pdf
    guidance.json
  models/
    maize_classifier.keras     <-- your real trained MobileNetV2 model
    labels.json               <-- original label order from training
```

If your original model filename is `agriguard_maize_v1.keras`, rename the copy to `maize_classifier.keras`. The model file is **not** included in this ZIP because it belongs to your Colab runtime. Do not download unknown third-party Keras files: load only your own trusted model.

### Optional semantic embeddings

```bash
python -m pip install sentence-transformers
```

Then check **Use pretrained multilingual semantic retrieval** in sidebar. First use downloads a pretrained model. Without it, TF-IDF still retrieves page-linked research excerpts. Semantic matching does not validate that recommendations are medically/agronomically appropriate.

## What the NLP feature does—and does not do

1. Select relevant paper pages by crop condition.
2. Rank research excerpts by the user's question.
3. Show verbatim source passage, PDF page, and DOI so a reviewer can verify them.
4. Display a separately curated management card.

**This is citation-backed retrieval, not an LLM producing free-text answers.** That deliberate first-version constraint reduces risk of inventing pesticide recommendations. A hosted LLM could be added after an agronomist reviews the source library and recommendation policies.

## Source caveats

Gidi's 2023 literature review covers common rust (pp. 2–3), maize lethal necrosis (pp. 3–4), gray leaf spot (pp. 4–5) and turcicum/northern leaf blight (pp. 5–6). Some recommendations mention pesticide products or application rates. These historical statements **are not validation of current Ethiopian registration or suitability**. The guidance cards intentionally omit chemical products and application rates.

MLN is **not an image classifier class**. Severe unknown symptoms require referral to qualified extension/plant-health services.

## Evidence, confidence and prototype claims

- Reported performance: 92.17% test accuracy on a **same-dataset random held-out split of 447 maize images**. External Ethiopian-field performance is unknown.
- Image softmax scores are not calibrated disease probabilities.
- The current `top_score < 0.70 OR margin < 0.20` rule is a **demonstration threshold**, not validated abstention.
- Incorrect or unfamiliar photos may still get high scores. For field use, add crop/quality gating, independent field tests, more conditions and clinical review.
- Guide users to inspect multiple plants and consult agricultural extension services before treatment choices.

## Demo checklist

1. Open Research Evidence **without** a model: confirm PDFs and page citations work.
2. Upload your trained `.keras` and `labels.json` into `models/`.
3. Upload a labeled independent maize photograph; see top scores and uncertainty.
4. Switch to Management Guidance. Check disease and citation.
5. Ask how to prevent the selected disease and inspect evidence with PDF page reference.
6. Export assessment JSON and evidence JSON.
7. Demonstrate uncertainty and study limitations explicitly.
