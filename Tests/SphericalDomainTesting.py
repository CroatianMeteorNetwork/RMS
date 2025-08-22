import RMS.Misc
import unittest

class TestSphericalDomain(unittest.TestCase):

    def test_nDimensionDomainSplit(self):
        # Test code - should produce no output
        test_data, results = [], []
        test_data.append([[160, 200, -180, 180], [80, 110, -90, +90], [23, 25, 0, 24]])
        results.append([[[160, 180], [80, 90], [23, 24]],
                        [[160, 180], [80, 90], [0, 1]],
                        [[160, 180], [-90, -70], [23, 24]],
                        [[160, 180], [-90, -70], [0, 1]],
                        [[-180, -160], [80, 90], [23, 24]],
                        [[-180, -160], [80, 90], [0, 1]],
                        [[-180, -160], [-90, -70], [23, 24]],
                        [[-180, -160], [-90, -70], [0, 1]]])

        test_data.append([[340, 350, 0, 360], [80, 110, -90, +90], [23, 25, 0, 24]])
        results.append([[[340, 350], [80, 90], [23, 24]],
                        [[340, 350], [80, 90], [0, 1]],
                        [[340, 350], [-90, -70], [23, 24]],
                        [[340, 350], [-90, -70], [0, 1]]])

        for test, expected_result in zip(test_data, results):
            result = RMS.Misc.nDimensionDomainSplit(test)
            self.assertEqual(result, expected_result)

    def test_domainWrapping(self):

        test_data = []
        test_data.append([[355, 362], [[355, 360], [0, 2]]])
        test_data.append([[-20, +40], [[340, 360], [0, 40]]])
        test_data.append([[+370, -10], [[370, -10]]])
        test_data.append([[+370, +380], [[10, 20]]])
        test_data.append([[+380, +370], [[380, 370]]])
        test_data.append([[+380, +10], [[380, 10]]])
        test_data.append([[+355, +360], [[355, 360]]])
        test_data.append([[-10, 0], [[350, 360]]])
        test_data.append([[359, 360], [[359, 360]]])
        test_data.append([[359 * 2, 360 * 2], [[358, 360]]])
        test_data.append([[0, 360], [[0, 360]]])

        for test in test_data:
            query_min, query_max = test[0][0], test[0][1]
            result = RMS.Misc.domainWrapping(query_min, query_max, 0, 360)
            self.assertEqual(result, test[1])


    def test_sphericalDomainWrapping(self):

        radec_test = []

        radec = [[[355, 365], [0, 10]],
                 [[[355, 360], [0, 10]], [[0, 5], [0, 10]]]]
        radec_test.append(radec)

        radec = [[[355, 365], [80, 100]],
                 [[[355, 360], [80, 90]], [[355, 360], [-90, -80]], [[0, 5], [80, 90]], [[0, 5], [-90, -80]]]]
        radec_test.append(radec)

        for test in radec_test:
            ra_min, ra_max = test[0][0][0], test[0][0][1]
            dec_min, dec_max = test[0][1][0], test[0][1][1]
            result = RMS.Misc.sphericalDomainWrapping(ra_min, ra_max, dec_min, dec_max)
            self.assertEqual(result, test[1])




if __name__ == '__main__':

    unittest.main()