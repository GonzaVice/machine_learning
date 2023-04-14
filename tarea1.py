# %% [markdown]
# IMPORTS BEFORE RUNNING CODE BELOW

# %%
import tensorflow.keras.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
import linear_models.softmaxreg as softmaxreg
import umap
import matplotlib.pyplot as plt
import numpy as np
import metrics.metrics as metrics

# %% [markdown]
# TRAINING MODEL USING THE IMAGES

# %%
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print ('{} {}'.format(x_train.shape, x_train.dtype))
print ('{} {}'.format(x_test.shape, x_train.dtype))
digit = x_train[0,:,:]
print(digit.shape)

# %%
#normalizamos/estandarizamos los datos
x_train_std = (x_train - x_train.mean()) / x_train.std()
x_test_std = (x_test - x_test.mean()) / x_test.std()

# %%
x_train_d_784 = x_train_std.reshape(60000,-1)
x_test_d_784 = x_test_std.reshape(10000,-1)

# %%
SM = softmaxreg.SoftmaxReg(10)

# %%
coeff = SM.fit(x_train_d_784, y_train)

# %%
y_pred = SM.predict(x_test_d_784)

# %%
acc = metrics.multiclass_accuracy(y_test, y_pred)
print(f"Accuracy: {acc}")

# %%
cm = metrics.confusion_matrix(y_test, y_pred, 10)
print(f'Confusion Matrix\n{cm}')

for row in cm:
    for col in row:
        print(col)

# %%
#2d visualization
reducer = umap.UMAP()
print('UMAP', flush = True)
reducer.fit(x_train_d_784) 
embedding = reducer.transform(x_train_d_784)
print(embedding.shape)
plt.scatter(embedding[:, 0], embedding[:, 1], c=y_train, cmap='Paired')
plt.show()

# %% [markdown]
# TRAINING MODEL USING HOG FUNCTION

# %%
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print ('{} {}'.format(x_train.shape, x_train.dtype))
print ('{} {}'.format(x_test.shape, x_train.dtype))
digit = x_train[0,:,:]
print(digit.shape)

# %%
#normalizamos/estandarizamos los datos
x_train_std = (x_train - x_train.mean()) / x_train.std()
x_test_std = (x_test - x_test.mean()) / x_test.std()
print(x_test_std.shape)
print(x_train_std.shape)

# %%
SM = softmaxreg.SoftmaxReg(10)

# %%
x_train_hog = list()
for n in range(len(x_train_std)):
    digit = x_train_std[n,:,:]
    fd = hog(digit, orientations=8, pixels_per_cell=(7,7), cells_per_block=(1, 1), visualize=False)
    x_train_hog.append(fd)
    if(n%9999 == 0): print(f"{round(n/len(x_train_std)*100)}%")
x_train_hog = np.array(x_train_hog)

# %%
x_test_hog = list()
for n in range(len(x_test_std)):
    digit = x_test_std[n,:,:]
    fd = hog(digit, orientations=8, pixels_per_cell=(7,7), cells_per_block=(1, 1), visualize=False)
    x_test_hog.append(fd)
    if(n%999 == 0): print(f"{round(n/len(x_test_std)*100)}%")
x_test_hog = np.array(x_test_hog)

# %%
coeff = SM.fit(x_train_hog, y_train)
y_pred = SM.predict(x_test_hog)

# %%
acc = metrics.multiclass_accuracy(y_test, y_pred)
print(f"Accuracy: {acc}")

# %%
cm = metrics.confusion_matrix(y_test, y_pred, 10)
print(f'Confusion Matrix\n{cm}')

# %%
#2d visualization
reducer = umap.UMAP()
print('UMAP', flush = True)
reducer.fit(x_train_hog) 
embedding = reducer.transform(x_train_hog)
print(embedding.shape)
plt.scatter(embedding[:, 0], embedding[:, 1], c=y_train, cmap='Paired')
plt.show()


