import unittest

from routing import HierarchicalRouter, RouteRequest, TopologySnapshot


class HierarchicalRouterTest(unittest.TestCase):
    def setUp(self):
        self.snapshot = TopologySnapshot(
            adjacency={0:[1],1:[0,2],2:[1,3,4],3:[2],4:[2,5],5:[4]},
            groups={10:[0,1],20:[2,3],30:[4,5]},
            node_group={0:10,1:10,2:20,3:20,4:30,5:30},
            weight=lambda _u,_v: 1,
        )

    def test_routes_across_two_layers(self):
        result = HierarchicalRouter().route(self.snapshot, RouteRequest(0,5))
        self.assertTrue(result.success)
        self.assertEqual(result.group_path,[10,20,30])
        self.assertEqual(result.path,[0,1,2,4,5])
        self.assertEqual(result.hops,4)

    def test_every_hop_is_a_real_link_and_path_has_no_loop(self):
        result = HierarchicalRouter().route(self.snapshot, RouteRequest(0,5))
        self.assertEqual(len(result.path),len(set(result.path)))
        self.assertTrue(all(v in self.snapshot.adjacency[u] for u,v in zip(result.path,result.path[1:])))

    def test_global_fallback_is_explicit(self):
        snapshot = TopologySnapshot({0:[1],1:[0,2],2:[1]}, {1:[0],2:[2]}, {0:1,2:2,1:1})
        result = HierarchicalRouter().route(snapshot, RouteRequest(0,2))
        self.assertTrue(result.success)
        self.assertEqual(result.fallback,"global_dijkstra")

    def test_unreachable_route_fails(self):
        snapshot = TopologySnapshot({0:[],1:[]},{1:[0],2:[1]},{0:1,1:2})
        self.assertFalse(HierarchicalRouter().route(snapshot,RouteRequest(0,1)).success)

    def test_all_inter_and_intra_algorithms_return_valid_paths(self):
        for inter in ("dijkstra_hops","dijkstra_weighted","greedy_best_first"):
            for intra in ("dijkstra_weighted","bfs_hops","bidirectional_bfs"):
                with self.subTest(inter=inter,intra=intra):
                    result=HierarchicalRouter().route(self.snapshot,RouteRequest(0,5,inter_algorithm=inter,intra_algorithm=intra))
                    self.assertTrue(result.success)
                    self.assertTrue(all(v in self.snapshot.adjacency[u] for u,v in zip(result.path,result.path[1:])))

    def test_learned_intra_router_is_used(self):
        calls=[]
        self.snapshot.learned_intra_router=lambda source,target,allowed,algorithm:(calls.append((source,target,algorithm)) or [0,1,2,4,5])
        for algorithm in ("per_action_d3qn","per_action_d3qn_beam"):
            result=HierarchicalRouter().route(self.snapshot,RouteRequest(0,5,intra_algorithm=algorithm))
            self.assertTrue(result.success)
        self.assertEqual([item[2] for item in calls],["per_action_d3qn","per_action_d3qn_beam"])


if __name__ == "__main__": unittest.main()
