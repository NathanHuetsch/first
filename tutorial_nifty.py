import nifty
import numpy as np
import sys

"""
# ------------------
# Initialize graph:
# ------------------
There are several ways to initialize a graph. 

1. The easiest is to build it manually:
    
    g = nifty.graph.UndirectedGraph(numberOfWantedNodes)
    g.insertEdge(node1, node2)
    ...
    
    
2. There is also an handy way to build a region-adjacency-graph from a label array representing 
the segmentation labels (both 2D-images or 3D-volumes are accepted):

    tiny_label_image = np.array([[0, 1, 3], 
                                 [0, 1, 4],
                                 [0, 2, 2]], dtype='uint32')
    rag = nifty.graph.rag.gridRag(tiny_label_image)
     
The created r.a.g. will already include edges between adjacent superpixels in the segmentation.
See online tutorial for more examples and more details about related functions:
http://derthorsten.github.io/nifty/docs/python/html/auto_examples/graph/plot_grid_graph_agglomerative_clustering.html#sphx-glr-auto-examples-graph-plot-grid-graph-agglomerative-clustering-py
http://derthorsten.github.io/nifty/docs/python/html/auto_examples/multicut/plot_isbi_2012_multicut_2D_simple.html#sphx-glr-auto-examples-multicut-plot-isbi-2012-multicut-2d-simple-py 


3. Then there is another special graph initialization function that already add edges 
between pixels connected by long-range relationships (like in the MWS algorithm).
 
"""

# Shape of the volume defining the 3D grid graph (10x50x50 pixels):
shape = [10,50,50]

# Offsets defines the 3D-connectivity patterns of the edges in the 3D pixel grid graph:
nb_local_offsets = 3
local_offsets = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]) # Local edges in three directions
# Long-range connectivity patterns:
non_local_offsets = np.array([[-1, -1, -1],
     [-1, 1, 1],
     [-1, -1, 1],
     [-1, 1, -1],
     [0, -9, 0],
     [0, 0, -9],
     [0, -9, -9],
     [0, 9, -9],
     [0, -9, -4],
     [0, -4, -9],
     [0, 4, -9],
     [0, 9, -4],
     [0, -27, 0],
     [0, 0, -27]])
offsets = np.concatenate((local_offsets, non_local_offsets))


is_local_offset = [True] * 3 + [False] * 14


# Build graph:
graph = nifty.graph.undirectedLongRangeGridGraph(np.array(shape), offsets,
                                                 is_local_offset=np.array(is_local_offset))
nb_nodes = graph.numberOfNodes
nb_edges = graph.numberOfEdges

# Get IDs of the local edges in the graph:
offset_index = graph.edgeOffsetIndex()
is_edge_local = np.zeros((nb_edges,), dtype='bool')
is_edge_local[offset_index < nb_local_offsets] = True

# Get weights of each edge:
affinities = np.random.uniform(size=shape + [offsets.shape[0]]) # Generate some random weights
edge_weights = graph.edges(affinities) # returned shape: (nb_edges, )

# ----------------
# Some additional fun stuff you can do:
# (examples of methods you could find useful)
# ----------------

# Get all uv IDs of the edges in the graph:
uvIds = graph.uvIds()
# Get nodes uv from edge ID:
u, v = graph.uv(4)
# Get edge ID from u-v nodes (return -1 if there is not such an edge in the graph):
edge_ID = graph.findEdge(u,v) # Vectorized version: graph.findEdges(array)

# Loop over the edges:
for e in graph.edges():
    pass

# iterate over all nodes
for node in graph.nodes():
    # iterate over all neigbours
    for otherNode, connectingEdge in graph.nodeAdjacency(node):
        pass




"""
# ---------------------------------
# Efficient implementation of the 
# UnionFind/DisjointSets kwargs-structure: 
# ---------------------------------
"""
import nifty.ufd as nufd

# Initialize UnionFind structure with 100 nodes:
UF = nufd.ufd(100)

# Find parent of node 3:
print(UF.find(3))

# Merge two nodes:
UF.merge(2, 3)

# Find parent of node 3:
print(UF.find(3))

# Find and merge can also accept vectorized inputs:
u_ids = np.arange(20,40,3) # some random pairs of nodes to merge
v_ids = np.arange(21,41,3)
uv_ids = np.stack([u_ids, v_ids], axis=-1)
UF.merge(uv_ids)

print(UF.find(uv_ids[:,1]))



"""
# ---------------------------------
# Efficient implementation of a graph  
# agglomeration/contraction: 
# ---------------------------------
"""

class MyCallback(nifty.graph.EdgeContractionGraphCallback):
    def __init__(self, number_of_edges=None):
        assert number_of_edges is not None
        # Initialize UnionFind structure for edges:
        self.uf_edges = nufd.ufd(number_of_edges)
        super(MyCallback, self).__init__()

    def contractEdge(self, edgeToContract):
        pass

    def mergeEdges(self, aliveEdge, deadEdge):
        self.uf_edges.merge(aliveEdge, deadEdge)

    def mergeNodes(self, aliveNode, deadNode):
        pass

    def contractEdgeDone(self, contractedEdge):
        pass



# the callback
callback = MyCallback(number_of_edges=nb_edges)

# Build the edge contraction graph:
contrGraph = nifty.graph.edgeContractionGraph(graph, callback)


# Contract some edges:
edges_to_contract = [1,4,56]


for e in edges_to_contract:

    # get the endpoints of e in the original
    u, v = graph.uv(e)

    # get the nodes in the contracted graph
    cu = contrGraph.findRepresentativeNode(u)
    cv = contrGraph.findRepresentativeNode(v)

    print("contraction graph:", cv)
    print("e", e, )
    print("      u", u, "cv", v)
    print("     cu", cu, " v", cv)

    if (cu != cv):
        # the edge is still alive
        # since the endpoints are still in
        # different clusters
        ce = contrGraph.findRepresentativeEdge(e)

        # lets contract that edge
        contrGraph.contractEdge(ce)

    else:
        # the edge is not alive any more
        tmp = contrGraph.nodeOfDeadEdge(e)
        assert tmp == cv



"""
# ---------------------------------
# ---------------------------------
# MULTICUT SOLVERS:
# ---------------------------------
# ---------------------------------
"""

# First, get some kind of undirected graph:
graph = nifty.graph.undirectedGraph(8)

# CNN usually predicts probabilities between 0 and 1.
# The costs in the multicut objective have instead positive (attractive) and negative (repulsive) values:
edge_costs = from_prob_to_cost(edge_weights)

# Build the MC objective:
mc_obj = graph.MulticutObjective(graph=graph, weights=edge_costs)

# ----------------------------------
# EXAMPLE A: solving the exact ILP problem (slow for big graphs)
log_visitor = mc_obj.loggingVisitor(verbose=True, timeLimitSolver=np.inf, timeLimitTotal=np.inf)
solverFactory = mc_obj.multicutIlpFactory()
solver = solverFactory.create(mc_obj)
final_node_labels_exact = solver.optimize(visitor=log_visitor)
# ----------------------------------


# ----------------------------------
# EXAMPLE B: use fusionMove solver
log_visitor = mc_obj.loggingVisitor(verbose=True, timeLimitSolver=np.inf, timeLimitTotal=np.inf)
# 1. Initialize a warm-up solver and run optimization
solverFactory = mc_obj.greedyAdditiveFactory()
solver = solverFactory.create(mc_obj)
node_labels = solver.optimize(visitor=log_visitor)
# 2. Use a second better warm-up solver to get a better solution:
solverFactory = mc_obj.kernighanLinFactory()
solver = solverFactory.create(mc_obj)
new_node_labels = solver.optimize(visitor=log_visitor, nodeLabels=node_labels)
# 3. Initialize the proposal needed for the FusionMoves:
pgen = mc_obj.watershedCcProposals(sigma=1.0, numberOfSeeds=0.1)
# 4. Run the funsionMuves solver
solverFactory = mc_obj.ccFusionMoveBasedFactory(proposalGenerator=pgen, numberOfIterations=100,
                                                        stopIfNoImprovement=10)
solver = solverFactory.create(mc_obj)
final_node_labels_fusion = solver.optimize(visitor=log_visitor, nodeLabels=new_node_labels)
# ----------------------------------


# Collect kwargs:
print(log_visitor.iterations())
print(log_visitor.energies())
print(log_visitor.runtimes())







# -------------------------------
# Logging and verbose visitors:
# they keep track of energy, running-time during optimization
# -------------------------------
from nifty import LogLevel
# .value("NONE", nifty::logging::LogLevel::NONE)
# .value("FATAL", nifty::logging::LogLevel::FATAL)
# .value("ERROR", nifty::logging::LogLevel::ERROR)
# .value("WARN", nifty::logging::LogLevel::WARN)
# .value("INFO", nifty::logging::LogLevel::INFO)
# .value("DEBUG", nifty::logging::LogLevel::DEBUG)
# .value("TRACE", nifty::logging::LogLevel::TRACE)

# visitNth defines how often we log
log_visitor = mc_obj.loggingVisitor(visitNth=1, verbose=True, timeLimitSolver=np.inf, timeLimitTotal=np.inf, logLevel=LogLevel.WARN)
log_visitor.stopOptimize()
print(log_visitor.iterations())
print(log_visitor.energies())
print(log_visitor.runtimes())

verb_visitor = mc_obj.verboseVisitor(visitNth=1, timeLimitSolver=np.inf, timeLimitTotal=np.inf, logLevel=LogLevel.WARN)
verb_visitor.stopOptimize()
print(verb_visitor.runtimeSolver)
print(verb_visitor.runtimeTotal)
print(verb_visitor.timeLimitSolver)
print(verb_visitor.timeLimitTotal)



# --------------------------------
# --------------------------------
# BUILDING THE SOLVER:
# --------------------------------
# --------------------------------


# --------------------------------
# Exact solvers:
# --------------------------------
"""
Create an instance of an ilp multicut solver.

        Find a global optimal solution by a cutting plane ILP solver
        as described in :cite:`Kappes-2011`
        and :cite:`andres_2011_probabilistic`


    Note:
        This might take very long for large models.

    Args:
        addThreeCyclesConstraints (bool) :
            explicitly add constraints for cycles
            of length 3 before opt (default: {True})
        addOnlyViolatedThreeCyclesConstraints (bool) :
            explicitly add all violated constraints for only violated cycles
            of length 3 before opt (default: {True})
        ilpSolverSettings (:class:`IlpBackendSettings`) :
            SettingsType of the ilp solver (default : {:func:`ilpSettings`})
        ilpSolver (str) : name of the solver. Must be in
            either "cplex", "gurobi" or "glpk".
            "glpk" is only capable of solving very small models.
            (default: {"cplex"}).

    Returns:
        %s or %s or %s : multicut factory for the corresponding solver
"""
solverFactory = mc_obj.multicutIlpFactory(addThreeCyclesConstraints=True,
                            addOnlyViolatedThreeCyclesConstraints=True,
                            ilpSolverSettings=None,
                            ilpSolver = None)
solver = solverFactory.create(mc_obj)
# solverFactory = mc_obj.multicutIlpCplexFactory()


# --------------------------------
# Warm-up solvers:
# they usually return solutions that are far away from being optimal,
# so they are usually used as warming-up solutions for better solvers like FusionMove
# --------------------------------
solverFactory = mc_obj.kernighanLinFactory(numberOfInnerIterations = sys.maxsize,
            numberOfOuterIterations = 100,
            epsilon = 1e-6)
solver = solverFactory.create(mc_obj)


solverFactory = mc_obj.greedyAdditiveFactory(weightStopCond=0.0, nodeNumStopCond=-1.0, visitNth=1)
solver = solverFactory.create(mc_obj)


"""
Find approximate solutions via
agglomerative clustering as in :cite:`TODO`.

    Note:
        This is just for comparison since it implements the
        same as :func:`greedyAdditiveFactory`.

    Args:
        numberOfInnerIterations (int): number of inner iterations (default: {sys.maxsize})
        numberOfOuterIterations (int): number of outer iterations        (default: {100})
        epsilon (float): epsilon   (default: { 1e-6})
        verbose (bool):                (default: {False})
        greedyWarmstart (bool): initialize with greedyAdditive  (default: {True})

"""
mc_obj.multicutAndresKernighanLinFactory(
            numberOfInnerIterations = sys.maxsize,
            numberOfOuterIterations = 100,
            epsilon = 1e-6,
            verbose = False,
            greedyWarmstart = False #initialize with greedyAdditive
            )

"""
This solver tries to decompose the model into
        sub-models  as described in :cite:`alush_2013_simbad`.
        If a model decomposes into components such that there are no
        positive weighted edges between the components one can
        optimize each model separately.



    Note:
        Models might not decompose at all.

    Args:
        submodelFactory: multicut factory for solving subproblems
            if model decomposes (default: {:func:`defaultMulticutFactory()`})
        fallthroughFactory: multicut factory for solving subproblems
            if model does not decompose (default: {:func:`defaultMulticutFactory()`})

"""
mc_obj.multicutDecomposerFactory(submodelFactory=None, fallthroughFactory=None)

# --------------------------------
# Fusion-move solver:
# --------------------------------

# First, build the proposal generator:


pgen = mc_obj.watershedCcProposals(sigma=1.0, numberOfSeeds=0.1)

# Other prop. generators:
mc_obj.watershedProposals(sigma=1.0, seedFraction=0.0)
mc_obj.greedyAdditiveProposals(sigma=1.0, weightStopCond=0.0, nodeNumStopCond=-1.0)
mc_obj.randomNodeColorCcProposals(numberOfColors=2)
mc_obj.interfaceFlipperCcProposals()


# Second, decide which solver should we use for the reduced problem: (here we solve the exact ILP)
# This is in general not necessary. If CPLEX is available, it will be automatically chosen.
fsMoveSett = mc_obj.fusionMoveSettings(mc_obj.multicutIlpCplexFactory())

# Finally, build the fusion move solver:
solverFactory = mc_obj.ccFusionMoveBasedFactory(proposalGenerator=pgen, numberOfIterations=100,
                                                        stopIfNoImprovement=10, numberOfThreads=8,
                                                fusionMove=fsMoveSett)


mc_obj.perturbAndMapSettings(numberOfIterations=1000,
                                numberOfThreads=-1,
                                verbose=1,
                                noiseType='normal',
                                noiseMagnitude=1.0,
                                mcFactory=None)


########################
# Cut - Glue - Cut
########################
mc_obj.cgcFactory(doCutPhase=True, doGlueAndCutPhase=True, mincutFactory=None,
            multicutFactory=None,
            doBetterCutPhase=False, nodeNumStopCond=0.1, sizeRegularizer=1.0)
