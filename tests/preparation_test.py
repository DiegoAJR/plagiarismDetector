from unittest import TestCase
from preparation import create_n_grams, create_embeddings, build_embeddings, preparation

class TestPreprocess(TestCase):
    
    def test_n_grams1(self):
        self.assertEqual({('The', 'quick', 'brown'), ('quick', 'brown', 'fox'), ('over', 'the', 'fence'), ('The', 'very', 'best'), ('brown', 'fox', 'jumps'), ('fox', 'jumps', 'around'), ('jumps', 'over', 'the'), ('jumps', 'around', 'the'), ('fox', 'jumps', 'over'), ('very', 'best', 'brown'), ('around', 'the', 'fence'), ('best', 'brown', 'fox')},
                         create_n_grams("The quick brown fox jumps over the fence".split(" "),"The very best brown fox jumps around the fence".split(" "),3))
    
    def test_n_grams2(self):
        self.assertEqual({('Close',), ('Open',), ('the',), ('door',)},
                         create_n_grams("Close the door".split(" "),"Open the door".split(" "),1))

    def test_create_embeddings1(self):
        self.assertEqual([[0, 0, 0], [0, 0, 0]],
                         create_embeddings(set(["the","nice","fox"]))) 

    def test_create_embeddings2(self):
        self.assertEqual([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                         create_embeddings(set(["holy","crab","I","am","hungry"])))
        
    def test_build_embeddings1(self):
        self.assertEqual([[1, 1, ], [1, 1, 1]],
                         build_embeddings([0,0,0,0],["the","nice","fox","the"],["the nice fox", "the nice bunny the"]))