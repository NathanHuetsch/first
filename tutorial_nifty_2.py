import nifty
import numpy as np
import sys

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
# ------------------
# Undirected graph:
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

"""
tiny_label_image = np.array([[0, 1, 3],
                             [0, 1, 4],
                             [0, 2, 2]], dtype='uint32')
graph = nifty.graph.rag.gridRag(tiny_label_image)
nb_nodes = graph.numberOfNodes
nb_edges = graph.numberOfEdges


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
# Solving the multicut problem:
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











"""
# ---------------------------------
# ---------------------------------
# MORE DETAILED LIST OF MULTICUT SOLVERS:
# (I am not sure you will need this for the project)
# ---------------------------------
# ---------------------------------
"""


# -------------------------------
# Logging and verbose visitors:
# they keep track of energy and running-time during optimization
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


# --------------------------------
# Other solvers:
# --------------------------------
settings_pertAndMap = mc_obj.perturbAndMapSettings(numberOfIterations=1000,
                                numberOfThreads=-1,
                                verbose=1,
                                noiseType='normal',
                                noiseMagnitude=1.0,
                                mcFactory=None)
mc_obj.perturbAndMap(mc_obj, settings_pertAndMap)

mc_obj.cgcFactory(doCutPhase=True, doGlueAndCutPhase=True, mincutFactory=None,
            multicutFactory=None,
            doBetterCutPhase=False, nodeNumStopCond=0.1, sizeRegularizer=1.0)