from unittest import TestCase
from preparation import preparation

class TestPreprocess(TestCase):
    
        
    def test_count_plagiarims(self):
        self.assertEqual(1,
        int(preparation([[[1]]], [[[1]]], "The day is so nice")))

    def test_count_plagiarims2(self):
        self.assertEqual(1,
        int(preparation([[[0.8]]], [[[1]]], "The beautiful and small dog is eating a lot of candy with his other dog friends")))

        
