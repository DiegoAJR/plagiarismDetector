
from unittest import TestCase
from preprocess import preprocessing

class TestPreparation(TestCase):
    
    def test_sentence(self):
        self.assertEqual(['the quick brown fox jumps over the fence'],
                         preprocessing('The quick brown fox jumps over the fence.',test=True))
        
    def test_special_chars(self):
        self.assertEqual(['a  0929 word'],
                         preprocessing('A %& 0929 word',test=True))
        
    def test_lowercase(self):
        self.assertEqual(['cambios que hay en mi'],
                         preprocessing('CAMBIOS QUE HAY EN MI',test=True))
        
    def test_split(self):
        self.assertEqual(['the dog is big', ' the cat is small'],
                         preprocessing('the dog is big. the cat is small',test=True))