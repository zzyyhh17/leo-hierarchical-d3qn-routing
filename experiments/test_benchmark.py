import unittest
from experiments.benchmark import compare_with_global_baseline
from routing import TopologySnapshot

class BenchmarkTest(unittest.TestCase):
    def test_uses_same_cost_for_baseline_and_hierarchical_route(self):
        snap=TopologySnapshot({0:[1],1:[0,2],2:[1]},{0:[0,1],1:[2]},{0:0,1:0,2:1},lambda _u,_v:2)
        row=compare_with_global_baseline(snap,[(0,2)])[0]
        self.assertTrue(row['success']); self.assertEqual(row['path_stretch'],1.0); self.assertEqual(row['baseline_hops'],2)

if __name__ == '__main__': unittest.main()
