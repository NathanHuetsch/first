import nifty.graph
import numpy as np
import matplotlib.pyplot as plt 

nodes = 5
graph = nifty.graph.undirectedGraph(nodes)

graph.insertEdge(0,1)
graph.insertEdge(0,2)

edges = np.array([[0,3],[1,2],[3,4],[1,4]])
graph.insertEdges(edges)
print(graph)

for node in graph.nodes():
    print("u", node)
    for v,e in graph.nodeAdjacency(node):
        print("v",v,"e",e)

for e in graph.edges():
    print("edge",e,"uv:",graph.uv(e))

nifty.graph.drawGraph(graph)
plt.show()