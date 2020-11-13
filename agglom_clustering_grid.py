import nifty.graph
import nifty.graph.agglo

import skimage.data
import skimage.segmentation

import vigra

import matplotlib.pyplot as plt 
import numpy as np
#%matplotlib auto
# double figure size
a,b = plt.rcParams["figure.figsize"]
plt.rcParams["figure.figsize"] = 2.0*a, 2.0*b

#load skimage coin image
img = skimage.data.coins().astype('float32')
shape = img.shape[0:2]
#print(img.shape)
#plt.imshow(img)
#plt.show()

#define the grid graph
grid = nifty.graph.undirectedGridGraph(shape)
#print(grid)

#edge weighted
interpixelShape = [2*s-1 for s in shape]

#vigra
tags = ['xy', 'xyz'][img.ndim-2] #2d or 3d
vigraImg = vigra.taggedView(img, tags)
imgBig = vigra.sampling.resize(vigraImg, interpixelShape)
edgeStrength = vigra.filters.gaussianGradientMagnitude(imgBig, 2.0)
edgeStrength = edgeStrength.squeeze()
edgeStrength = np.array(edgeStrength)
gridGraphEdgeStrength = grid.imageToEdgeMap(edgeStrength, mode='interpixel')

#agglom clustering
edgeSizes = np.ones(grid.edgeIdUpperBound+1)
nodeSizes = np.ones(grid.nodeIdUpperBound+1)

clusterPolicy = nifty.graph.agglo.edgeWeightedClusterPolicy(
    graph = grid, edgeIndicators = gridGraphEdgeStrength,
    edgeSizes = edgeSizes, nodeSizes= nodeSizes, 
    numberOfNodesStop = 25, sizeRegularizer = 0.3)

agglomerativeClustering = nifty.graph.agglo.agglomerativeClustering(clusterPolicy)
agglomerativeClustering.run()
seg = agglomerativeClustering.result()
seg = seg.reshape(shape)

#plot segmentation
b_img = skimage.segmentation.mark_boundaries(img/255,
        seg.astype('uint32'), mode='inner', color=(0.1,0.1,0.2))

f = plt.figure()
f.add_subplot(1,3,1)
plt.imshow(img)
plt.title("img")

f.add_subplot(1,3,2)
plt.imshow(seg)
plt.title('Label seg')

f.add_subplot(1,3,3)
plt.imshow(b_img)
plt.title('segmentation')

plt.show()