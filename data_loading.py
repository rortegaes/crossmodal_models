import requests
import h5py
import numpy as np

def file_download():
    '''print("LOADING DATASETS AND MODEL WEIGHTS")

    title_abstract_2clusters = "title_abstract_2clusters.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(title_abstract_2clusters, 'wb').write(r.content)

    title_abstract_2clusters_weights = "title_abstract_2clusters_weights.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(title_abstract_2clusters_weights, 'wb').write(r.content)

    title_abstract_5class = "title_abstract_5class.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(title_abstract_5class, 'wb').write(r.content)

    title_abstract_5class_weights = "title_abstract_5class_weights.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(title_abstract_5class_weights, 'wb').write(r.content)

    print("BASELINE DATA LOADED") 

    captions_2clusters = "captions_2clusters.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(captions_2clusters, 'wb').write(r.content)

    captions_2clusters_weights = "captions_2clusters_weights.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(captions_2clusters_weights, 'wb').write(r.content)

    captions_5class = "captions_5class.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(captions_5class, 'wb').write(r.content)

    captions_5class_weights = "captions_5class_weights.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(captions_5class_weights, 'wb').write(r.content)

    print("EXPERIMENT #1 DATA LOADED") 

    figures_2clusters = "figures_2clusters.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(figures_2clusters, 'wb').write(r.content)

    figures_2clusters_weights = "figures_2clusters_weights.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(figures_2clusters_weights, 'wb').write(r.content)

    figures_5class = "figures_5class.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(figures_5class, 'wb').write(r.content)

    figures_5class_weights = "figures_5class_weights.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(figures_5class_weights, 'wb').write(r.content)

    print("EXPERIMENT #2 DATA LOADED") 

    cross = "cross.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(cross, 'wb').write(r.content)

    cross_weights = "cross_weights.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(cross_weights, 'wb').write(r.content)

    print("EXPERIMENT #3 DATA LOADED") 

    captions_2clusters_cross_weights = "captions_2clusters_cross_weights.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(captions_2clusters_cross_weights, 'wb').write(r.content)

    captions_5class_cross_weights = "captions_5class_cross_weights.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(captions_5class_cross_weights, 'wb').write(r.content)

    print("EXPERIMENT #4 DATA LOADED") 

    figures_2clusters_cross_weights = "figures_2clusters_cross_weights.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(figures_2clusters_weights, 'wb').write(r.content)

    figures_5class_cross_weights = "figures_5class_cross_weights.h5"
    url = #URL ZENODO
    r = requests.get(url, allow_redirects=True)
    open(figures_5class_weights, 'wb').write(r.content)

    print("EXPERIMENT #5 DATA LOADED") '''

def gen_text (h5path, indices,batchSize, shuffle): 
  db = h5py.File(h5path, "r")
  while True:
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, len(indices), batchSize):
        batch_indices = indices[i:i+batchSize]
        batch_indices.sort()
        
        bx = db["text"][batch_indices,:]
        by = db["labels"][batch_indices,:]

        yield (bx, by)

def gen_images (h5path, indices,batchSize, shuffle):
  db = h5py.File(h5path, "r")
  while True:
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, len(indices), batchSize):
        batch_indices = indices[i:i+batchSize]
        batch_indices.sort()
        
        bx = db["images"][batch_indices,:,:,:]
        by = db["labels"][batch_indices,:]

        yield (bx, by)

def gen_cross (h5path, indices,batchSize,shuffle):
  db = h5py.File(h5path, "r")
  while True:
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, len(indices), batchSize):
        batch_indices = indices[i:i+batchSize]
        batch_indices.sort()
        
        bx1 = db["text"][batch_indices,:]
        bx2 = db["images"][batch_indices,:,:,:]
        by = db["labels"][batch_indices,:]

        yield ([bx1, bx2], by)

