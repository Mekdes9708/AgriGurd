"""AgriGuard AI: maize image screening + cited agricultural research retrieval."""
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image
from rag import retrieve, load_passages, coverage_flag
from vision import load_labels, predict

BASE = Path(__file__).parent
MODEL_PATH = BASE/'models'/'maize_classifier.keras'
GUIDANCE = json.loads((BASE/'knowledge'/'guidance.json').read_text(encoding='utf-8'))
DISPLAY = {x:x.replace('_',' ').title() for x in GUIDANCE}
DISPLAY['healthy'] = 'Healthy appearance (limited classes)'

st.set_page_config(page_title='AgriGuard AI - Maize Health', page_icon='🌽', layout='wide')
st.title('🌽 AgriGuard AI')
st.caption('Experimental maize condition screening + evidence-linked Ethiopian crop guidance')
st.warning('Research prototype. A photo cannot confirm a disease, rule out other problems, or justify pesticide use. Ask a trained agricultural extension officer when uncertain.')

@st.cache_resource(show_spinner=False)
def get_model():
    from tensorflow import keras
    # Only load a trusted model that you trained and exported yourself.
    return keras.models.load_model(MODEL_PATH, compile=False, safe_mode=True)

@st.cache_resource(show_spinner=False)
def get_semantic_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

with st.sidebar:
    st.header('Assessment settings')
    crop_stage = st.selectbox('Crop stage', ['Not sure', 'Seedling', 'Vegetative', 'Flowering / grain filling'])
    spread = st.selectbox('Where are symptoms?', ['Not sure', 'One or a few plants', 'Several plants across the field'])
    st.divider()
    st.header('Research retrieval')
    use_embeddings = st.checkbox('Use pretrained multilingual semantic retrieval', value=False,
        help='Requires sentence-transformers and initial model download. TF-IDF works offline after library installation.')
    question = st.selectbox('What does the farmer need help with?', [
        'How do I recognize the signs?', 'What should I do now?', 'How can I prevent this next season?', 'What evidence supports these suggestions?'
    ])
    st.caption('Source: Gidi (2023), an Ethiopian maize disease review. Not a substitute for current approved extension guidance.')

photo, guide, research = st.tabs(['📷 Photo screening', '🌱 Management guidance', '📖 Research evidence'])

with photo:
    left,right=st.columns([1,1])
    with left:
        mode=st.radio('Select image source', ['Upload photo', 'Use camera'], horizontal=True)
        upload = st.file_uploader('Upload JPG or PNG maize leaf', type=['jpg','jpeg','png']) if mode == 'Upload photo' else st.camera_input('Take a maize leaf photograph')
        if upload is not None:
            raw=upload.getvalue()
            if len(raw) > 12_000_000:
                st.error('Image exceeds 12 MB. Choose a smaller image.'); st.stop()
            try:
                img = Image.open(io.BytesIO(raw))
                img.verify()
                img = Image.open(io.BytesIO(raw)).convert('RGB')
                if img.width*img.height > 24_000_000:
                    st.error('Image dimensions are too large.'); st.stop()
                st.image(img, caption='Your submitted maize photograph', use_container_width=True)
            except Exception as exc:
                st.error('Image could not be opened: '+str(exc)); st.stop()
    with right:
        if not MODEL_PATH.exists():
            st.info('The trained classifier is not installed yet. Copy your Colab model and labels.json into the models folder. You can still explore research guidance below.')
        elif upload is not None:
            try:
                labels = load_labels()
                result = predict(get_model(), img, labels)
                st.session_state['ag_result'] = result
                st.session_state['ag_stage'] = crop_stage
                st.session_state['ag_spread'] = spread
            except Exception as exc:
                st.error(f'Model prediction unavailable: {exc}')
        current = st.session_state.get('ag_result') if upload is not None else None
        if current:
            st.subheader('Possible condition: '+DISPLAY[current['label']])
            st.metric('Top model score',f"{current['top_score']:.1%}")
            st.caption('Softmax score, not calibrated diagnostic confidence.')
            if current['uncertain']:
                st.error('Uncertain screening result: retake photos or ask an extension worker. Do not act on this label alone.')
            else:
                st.warning('Provisional model match only. Unknown diseases and stress may be misclassified.')
            for name,score in current['ranks']:
                st.write(f'**{DISPLAY[name]}** — {score:.1%}')
                st.progress(float(score))
            payload = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), crop='maize',
                model_version='MobileNetV2 v1 (same-dataset test accuracy 92.17%)',
                crop_stage=crop_stage, affected_plants=spread, prediction=current,
                status='Experimental; needs expert confirmation')
            st.download_button('Download assessment JSON',json.dumps(payload,indent=2),
                               file_name='agriguard_assessment.json',mime='application/json')
        elif upload is None:
            st.info('Upload a maize photograph to see the model screening result.')

with guide:
    current=st.session_state.get('ag_result')
    default_condition=current['label'] if current else 'gray_leaf_spot'
    select=st.selectbox('Research condition (select independently of model output)', list(GUIDANCE),
                        index=list(GUIDANCE).index(default_condition), format_func=lambda x: DISPLAY[x])
    g=GUIDANCE[select]
    st.subheader(g['title'])
    st.caption(g['scientific_name'])
    st.write('**Symptom checks**')
    st.write(g['confirm'])
    st.write('**Practical next steps**')
    for item in g['precautions']:st.write('• '+item)
    st.write('**For future seasons**')
    for item in g['prevention']:st.write('• '+item)
    st.info('Limitations: '+g['limitations'])
    if g['citations']:
        st.caption('Guidance reviewed against: '+', '.join(g['citations']))
    st.caption('We intentionally exclude pesticide products, dosing, and chemical application recommendations until local registration and professional approval are verified.')

with research:
    current=st.session_state.get('ag_result')
    choice=st.selectbox('Search condition', ['gray_leaf_spot','common_rust','northern_leaf_blight','maize_lethal_necrosis'],
        index=['gray_leaf_spot','common_rust','northern_leaf_blight','maize_lethal_necrosis'].index(current['label']) if current and current['label'] in ['gray_leaf_spot','common_rust','northern_leaf_blight','maize_lethal_necrosis'] else 0,
        format_func=lambda x: x.replace('_',' ').title())
    extra=st.text_input('Research question / search terms', value=question)
    with st.spinner('Finding source excerpts...'):
        semantic=None
        if use_embeddings:
            try:semantic=get_semantic_model()
            except Exception as exc:st.error(f'Embedding model unavailable: {exc}. Switch back to TF-IDF.');st.stop()
        results=retrieve(extra,choice,k=5,model=semantic)
    st.info(coverage_flag(results))
    st.caption('Evidence is quoted from the uploaded Ethiopian research review with original PDF page numbers. Similarity measures retrieval ranking only, not the quality of management advice.')
    for item in results:
        with st.expander(f"{item['source']} · PDF page {item['page']} · retrieval {item['score']:.2f}", expanded=False):
            st.write(item['text'])
            st.markdown(f"[Open publication DOI]({item['url']})")
            st.caption(f"Evidence ID: {item['id']}")
    st.download_button('Download retrieved evidence JSON',
        json.dumps({'condition':choice,'question':extra,'excerpts':results},indent=2),
        file_name='agriguard_research_excerpts.json',mime='application/json')
    st.divider()
    st.subheader('Study and safety notes')
    st.write('This prototype performs source retrieval, not unrestricted LLM answer generation. The curated guidance above is the safe offline fallback. The retrieved source contains historical pesticide mentions that are not verified as currently registered in Ethiopia; do not act on them without local professional advice.')
    st.write('The image model currently supports just four classes; maize lethal necrosis is included here only for research and referral, not image prediction.')
