# Getting an already trained model
import  tensorflow as tf
from os.path import join

from sklearn.decomposition import PCA
import pylab as plt


modelsFolder = join('G:/','IA','Models')
proust =tf.keras.models.load_model(join(modelsFolder, 'temps_perdu'))

proust.summary()


#DONT FORGET TO SAVE THE VOCAB -__-'
e = proust.layers[0]
wghts = e.get_weights()[0]
print(wghts.shape) 


# Reduce the dim


pca = PCA(n_components=3)
pca.fit(wghts)
lowDim = pca.transform(wghts)


# Display
import pylab as plt

fig = plt.figure()
ax = fig.gca(projection='3d')
for idx, v in enumerate(lowDim):
    ax.text(v[0], v[1], v[2],   idx2word[idx])



ax.set_xlim(lowDim.min(), lowDim.max())
ax.set_ylim(lowDim.min(), lowDim.max())
ax.set_zlim(lowDim.min(), lowDim.max())
plt.show()
