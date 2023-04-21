
from unittest import TestCase
from preprocess import preprocessing

class TestPreparation(TestCase):
    
    def test_sentence(self):
        self.assertEqual("the quick brown fox jump fence",
                         preprocessing('The quick brown fox jumps over the fence.',test=True))
        
    def test_special_chars(self):
        self.assertEqual("a  0929 word",
                         preprocessing('A %& 0929 word',test=True))
        
    def test_lemmatization(self):
        self.assertEqual("job stop hardrock fox panda",
                         preprocessing('jobs stops hardrock foxes pandas',test=True))
        
    def test_lowercase(self):
        self.assertEqual("cambios que hay en mi",
                         preprocessing('CAMBIOS QUE HAY EN MI',test=True))