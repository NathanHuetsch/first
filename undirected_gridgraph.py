import nifty.graph
import matplotlib.pyplot as plt 

shape = [3,3]
graph = nifty.graph.undirectedGridGraph(shape)
nifty.graph.drawGraph(graph)
plt.show()




for node in graph.nodes():
    print("node",node,"coordiante",graph.nodeToCoordinate(node))

