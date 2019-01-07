from main_lucas import *
from HKS import *

#test1
V, _, T = loadOffFile("Data/FAUST_002.off") ## loading the data with the parser
n = 100
#t = np.array([5,10,50, 100, 200])
t = [0.01, 0.1]
hks = getHKS(V, T, n, np.array(t))

#enclose in fct
def build_graph(V,T,hks):

    Gf = nx.Graph()
    for i in range(len(V)):
        Gf.add_node(i, value=hks[i])
    
    for t in T:
        Gf.add_edge(t[0], t[1])
        Gf.add_edge(t[1], t[2])
        Gf.add_edge(t[2], t[0])

    return Gf

def compute_diags(filename, t):
    V, _, T = loadOffFile(filename) ## loading the data with the parser
    n = 100
    hks = getHKS(V, T, n, np.array(t))
    for f in hks.T:
        Gf = build_graph(V,T,f)
        pers_pts = persistence_pts_from_Gf(Gf)
        plt.scatter(pers_pts[:,1], pers_pts[:,0])
    return

#test on several meshes with HKS on different scales
compute_diags("Data/human1.off", [0.001, 0.01,0.1,1.,10.,100.])
compute_diags("Data/FAUST_002.off", [0.001, 0.01,0.1,1.,10.,100.])
compute_diags("Data/FAUST_006.off", [0.001, 0.01,0.1,1.,10.,100.])
