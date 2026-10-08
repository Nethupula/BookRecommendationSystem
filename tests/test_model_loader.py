import io
import unittest
import zipfile
from pathlib import Path

import h5py
import numpy as np

from src.model_loader import load_recommender

MODEL=Path(__file__).resolve().parents[1]/'models/neural_recommender.keras'


@unittest.skipUnless(MODEL.exists(),'Local neural artifact required')
class LoaderTests(unittest.TestCase):
    def test_preserves_every_learned_weight_and_rating_scale(self):
        model=load_recommender(MODEL)
        checked=0
        with zipfile.ZipFile(MODEL) as archive, h5py.File(io.BytesIO(archive.read('model.weights.h5')),'r') as weights:
            for group in weights['layers'].values():
                saved=group['vars']
                if not len(saved):
                    continue
                layer=model.get_layer(saved.attrs['name'])
                loaded=layer.get_weights()
                for index,key in enumerate(sorted(saved.keys(),key=int)):
                    np.testing.assert_array_equal(loaded[index],saved[key][...])
                    checked+=1
        self.assertEqual(checked,8)
        scale=model.get_layer('rating_scale')
        np.testing.assert_allclose(scale(np.array([[0.],[.5],[1.]],dtype=np.float32)).numpy(),[[1.],[5.5],[10.]])
        output=model.predict([np.zeros((3,1),dtype=np.int32),np.arange(3,dtype=np.int32).reshape(-1,1)],verbose=0)
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(((output>=1)&(output<=10)).all())


if __name__=='__main__':
    unittest.main()
