import nifty
import numpy
import inspect

source = inspect.getsource(nifty.graph.randomGraph)
#print(source)
source_file = inspect.getsourcefile(numpy.stack)
#inspect.getsourcefile(nifty.graph.randomGraph)
print(source_file)