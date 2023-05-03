from unittest import TestCase
from preparation import preparation

class TestPreprocess(TestCase):
    
        
    def test_count_plagiarims(self):
        self.assertEqual(5,
        preparation(["The day is so nice"], ["The day is so nice"]))

    def test_count_plagiarims2(self):
        self.assertEqual(16,
        preparation(["The beautiful and small dog is eating a lot of candy with his other dog friends"], ["The cute and tiny dog and his friends are eating a ton of sweets"]))
        
