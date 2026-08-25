import unittest
from routing.metrics import evaluate_path

class MetricsTest(unittest.TestCase):
    def test_shared_route_metrics(self):
        weights={(0,1):100.0,(1,2):200.0}
        result=evaluate_path([0,1,2],lambda u,v:weights[(u,v)],baseline_cost=250)
        self.assertEqual(result.hops,2)
        self.assertEqual(result.distance_km,300)
        self.assertAlmostEqual(result.path_stretch,1.2)
        self.assertAlmostEqual(result.propagation_latency_ms,300/299792.458*1000)

if __name__ == '__main__': unittest.main()
