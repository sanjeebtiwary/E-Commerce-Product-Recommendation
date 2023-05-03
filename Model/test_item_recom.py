import unittest
import pandas as pd
from recommendation_engine import RecommendationEngine


class TestRecommendationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RecommendationEngine('test_data.csv')

    def test_find_similar_users(self):
        similarities, indices = self.engine.find_similar_users('1')
        self.assertAlmostEqual(similarities[0], 1.0)
        self.assertListEqual(indices[0].tolist(), [0, 2, 1])

    def test_find_similar_items(self):
        similarities, indices = self.engine.find_similar_items('A')
        self.assertAlmostEqual(similarities[0], 1.0)
        self.assertListEqual(indices[0].tolist(), [0, 1, 2])

    def test_predict_userbased(self):
        prediction = self.engine.predict_userbased('2', 'B')
        self.assertEqual(prediction, 4)

    def test_predict_itembased(self):
        prediction = self.engine.predict_itembased('2', 'B')
        self.assertEqual(prediction, 4)


if __name__ == '__main__':
    unittest.main()
