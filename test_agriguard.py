import io
import json
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from rag import load_passages, retrieve, coverage_flag
from vision import preprocess, predict, DEFAULT_LABELS

class MockModel:
    def __init__(self, x):self.x=np.array([x],dtype=float)
    def predict(self, a, verbose=0):return self.x

class Tests(unittest.TestCase):
    def test_pdf_excerpts_include_page(self):
        ps=load_passages()
        self.assertGreater(len(ps),10)
        self.assertTrue(all(1<=x['page']<=6 for x in ps))
    def test_retrieval_gray_spot(self):
        x=retrieve('gray leaf spot rectangular lesions parallel veins', 'gray_leaf_spot')
        self.assertTrue(len(x)>0)
        self.assertIn(x[0]['page'], (4,5))
        self.assertIn('GIDI2023',x[0]['id'])
    def test_retrieval_common_rust(self):
        x=retrieve('puccinia sorghi common rust pustules symptoms', 'common_rust')
        self.assertTrue(len(x)>0)
        self.assertIn(x[0]['page'], (2,3))
    def test_healthy_no_research(self):
        self.assertEqual(retrieve('any question', 'healthy'),[])
    def test_no_hallucinated_fallback(self):
        self.assertIn('Do not invent',coverage_flag([]))
    def test_image_preprocess(self):
        im=Image.new('RGB',(90,300),'green')
        x=preprocess(im)
        self.assertEqual(x.shape,(1,224,224,3))
        self.assertEqual(x.dtype,np.float32)
    def test_scores_and_margin(self):
        m=MockModel([0.03,0.04,0.10,0.83])
        result=predict(m,Image.new('RGB',(224,224)),DEFAULT_LABELS)
        self.assertEqual(result['label'],'common_rust')
        self.assertFalse(result['uncertain'])
    def test_uncertain(self):
        result=predict(MockModel([.25,.25,.25,.25]),Image.new('RGB',(224,224)))
        self.assertTrue(result['uncertain'])
    def test_bad_softmax(self):
        with self.assertRaises(ValueError):predict(MockModel([2,0,0,0]),Image.new('RGB',(224,224)))
    def test_guidance_citations(self):
        g=json.loads((Path(__file__).parent/'knowledge'/'guidance.json').read_text())
        for c in DEFAULT_LABELS:
            self.assertIn(c,g)
            self.assertTrue(g[c]['confirm'])
            self.assertTrue(g[c]['precautions'])
        self.assertEqual(g['healthy']['citations'],[])

if __name__=='__main__':unittest.main()
