import unittest
from routing.grouping import connected_groups

class GroupingTest(unittest.TestCase):
    def test_assigns_every_node_once_with_size_limit(self):
        adjacency={i:[j for j in (i-1,i+1) if 0<=j<12] for i in range(12)}
        groups,node_group=connected_groups(adjacency,5)
        self.assertEqual(set(node_group),set(adjacency))
        self.assertTrue(all(len(nodes)<=5 for nodes in groups.values()))
        self.assertEqual(sum(map(len,groups.values())),len(adjacency))

if __name__ == '__main__': unittest.main()
