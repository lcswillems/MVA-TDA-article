import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.misc as misc
from scipy.ndimage import distance_transform_cdt
import networkx as nx

# In the rest of the program, a graph function is a networkx graph
# where a value is associated to each node.

class UnionFind:
    """Slightly modified version of networkx.utils.union_find."""

    def __init__(self):
        """Create a new empty union-find structure."""

        self.weights = {}
        self.parents = {}

    def __setitem__(self, object, value):
        """Set the weight of the root of the object."""

        root = self[object]
        self.weights[root] = value

    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""

        # check for previously unknown object
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = float("-inf")
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""

        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r],r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.parents[r] = heaviest

def persistence_pts_from_Gf(Gf):
    """Return the points of the persistence diagram of the given
    graph function. It uses the algorithm described in slide 107 of
    http://graphics.stanford.edu/courses/cs468-17-spring/LectureSlides/2017CS468_05_31_v1.pdf."""

    def get_value(x):
        return Gf.nodes[x]['value']

    def mark(x):
        Gf.nodes[x]['marked'] = True

    def is_marked(x):
        if 'marked' in Gf.nodes[x].keys():
            return Gf.nodes[x]['marked']
        return False

    def get_marked_neighbors(x):
        return [nx for nx in Gf.neighbors(x) if is_marked(nx)]

    union_find = UnionFind()
    decreasing_nodes = sorted(Gf.nodes, key=lambda x: Gf.nodes[x]['value'])
    persistence_dict = {}

    for x in decreasing_nodes:
        v = get_value(x)
        marked_neighbors = get_marked_neighbors(x)

        if len(marked_neighbors) == 0:
            union_find[x] = v
            persistence_dict[x] = {'birth': v}
        else:
            roots = list({union_find[x] for x in marked_neighbors})

            heaviest = max([(union_find.weights[r],r) for r in roots])[1]
            for r in roots:
                if r != heaviest:
                    persistence_dict[r]['death'] = v

            nodes = marked_neighbors + [x]
            union_find.union(*nodes)
        mark(x)
    persistence_dict[union_find[x]]['death'] = v

    persistence_pts = [(v['birth'], v['death']) for v in persistence_dict.values()]
    persistence_pts = np.array(persistence_pts)
    return persistence_pts