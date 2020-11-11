import nifty.cgp as ncgp
import nifty.segmentation as nseg

import skimage.data as sdata
import skimage.filters as sfilt
import numpy as np

import matplotlib.pyplot as plt 

#create data
img = sdata.coins()
#edge indicator
edgeIndicator = sfilt.prewitt(sfilt.gaussian(img, 3))
#watershed oversegmentation
seeds = nseg.localMaxima(edgeIndicator)
print(seeds)
#overseg = nseg.seededWatersheds(edgeIndicator,seeds=seeds, method='node_weighted', acc='min')
#plt.imshow(overseg)
