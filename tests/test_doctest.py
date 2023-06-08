import doctest
import unittest

from spacebench.datamaster import DataMaster



def test_doctest_suit():
    # test_suit = unittest.TestSuite()
    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner()

    # add tests
    for test in finder.find(DataMaster):
        runner.run(test)
    # test_suit.addTest(doctest.DocTestSuite(DataMaster))
    
    # set runner
    # runner = unittest.TextTestRunner(verbosity=2).run(test_suit)

    assert not runner.failures