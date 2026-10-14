
import numpy as np
import uproot
import awkward as ak
# Load the array
file="training_data_CMS/y_test.npy"
data = np.load(file, mmap_mode="r")
data = uproot.open("training_data_CMS/data_chunk_457.root")
print(data.keys())

# Extract a tree directly into a Pandas DataFrame
tree = data["data;1"]
df = tree.arrays(library="ak")
print(df.shape)
# Inspect metadata and data
# print(file)
# print("Shape:", data.shape)
# print("Data Type:", data.dtype)
# print("Is Fortran order:", data.flags["F_CONTIGUOUS"])
# first_slice = data[0]
#print("first_slice =", repr(first_slice))

#X_train = np.load('training_data_CMS/X_train.npy')
#y_train = np.load('training_data_CMS/y_train.npy')

# To run code on pixi
# /software/mm24721/jet_tagger_BSc/.pixi/envs/default/bin/python test.py