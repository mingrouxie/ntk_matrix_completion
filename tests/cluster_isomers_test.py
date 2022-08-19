import pathlib
import sys
import pdb

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))

from utils.utilities import cluster_isomers
import unittest


class TestClusterIsomers(unittest.TestCase):
    def cases(self):
        return [
            {
                "input_smiles": [
                    "C[N@@+]1(Cc2ccccc2)CCCC[C@@H]1CCC[C@H]1CCCC[N@@+]1(C)Cc1ccccc1",
                    "C[N@+]1(Cc2ccccc2)CCCC[C@H]1CCC[C@H]1CCCC[N@+]1(C)Cc1ccccc1",
                    "C[N@@+]1(Cc2ccccc2)CCCC[C@H]1CCC[C@@H]1CCCC[N@@+]1(C)Cc1ccccc1",
                    "C[N@+]1(Cc2ccccc2)CCCC[C@@H]1CCC[C@H]1CCCC[N@+]1(C)Cc1ccccc1",
                    "C[C@@H]1C[N@+]2(CCCC[N@+]34CCN(C[C@H]3C)[C@H](C)C4)CCN1C[C@H]2C",
                    "C[N@@+]1(Cc2ccccc2)CCCC[C@H]1CCC[C@H]1CCCC[N@@+]1(C)Cc1ccccc1",
                    "C[C@H]1C[N@+]2(CCCC[N@+]34CCN(C[C@H]3C)[C@H](C)C4)CCN1C[C@@H]2C",
                    "C[C@H]1C[N@+]2(CCCC[N@@+]34CCN(C[C@H]3C)[C@@H](C)C4)CCN1C[C@H]2C",
                    "C[C@H]1C[N@+]2(CCCC[N@+]34CCN(C[C@H]3C)[C@@H](C)C4)CCN1C[C@@H]2C",
                    "C[C@H]1C[N@+]2(CCCC[N@+]34CCN(C[C@@H]3C)[C@H](C)C4)CCN1C[C@@H]2C",
                ],
                "expected_return": {
                    "C[N+]1(Cc2ccccc2)CCCCC1CCCC1CCCC[N+]1(C)Cc1ccccc1": {
                        "C[N@@+]1(Cc2ccccc2)CCCC[C@H]1CCC[C@H]1CCCC[N@@+]1(C)Cc1ccccc1",
                        "C[N@+]1(Cc2ccccc2)CCCC[C@H]1CCC[C@H]1CCCC[N@+]1(C)Cc1ccccc1",
                        "C[N@+]1(Cc2ccccc2)CCCC[C@@H]1CCC[C@H]1CCCC[N@+]1(C)Cc1ccccc1",
                        "C[N@@+]1(Cc2ccccc2)CCCC[C@H]1CCC[C@@H]1CCCC[N@@+]1(C)Cc1ccccc1",
                        "C[N@@+]1(Cc2ccccc2)CCCC[C@@H]1CCC[C@H]1CCCC[N@@+]1(C)Cc1ccccc1",
                    },
                    "CC1C[N+]2(CCCC[N+]34CCN(CC3C)C(C)C4)CCN1CC2C": {
                        "C[C@H]1C[N@+]2(CCCC[N@+]34CCN(C[C@@H]3C)[C@H](C)C4)CCN1C[C@@H]2C",
                        "C[C@@H]1C[N@+]2(CCCC[N@+]34CCN(C[C@H]3C)[C@H](C)C4)CCN1C[C@H]2C",
                        "C[C@H]1C[N@+]2(CCCC[N@+]34CCN(C[C@H]3C)[C@H](C)C4)CCN1C[C@@H]2C",
                        "C[C@H]1C[N@+]2(CCCC[N@@+]34CCN(C[C@H]3C)[C@@H](C)C4)CCN1C[C@H]2C",
                        "C[C@H]1C[N@+]2(CCCC[N@+]34CCN(C[C@H]3C)[C@@H](C)C4)CCN1C[C@@H]2C",
                    },
                },
            },
        ]

    def test_cluster_isomers(self):
        for test_case in self.cases():
            self.assertEqual(
                cluster_isomers(
                    test_case["input_smiles"],
                ),
                test_case["expected_return"],
            )


if __name__ == "__main__":
    unittest.main()
