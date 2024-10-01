import unittest
from ml_code.data_load import load_data

class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        data = load_data('E:/Masters/Maryland/Sem1/ENPM611 Software Engineering/ml-project/data/data.csv')
        self.assertIsNotNone(data)
        self.assertGreater(len(data),150)
# python -m unittest test.unit_test     
if __name__ == '__main__':
    unittest.main()