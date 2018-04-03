import os

path=os.getcwd()
filenames=os.listdir(path)

for filename in filenames:
    os.rename(filename, filename.replace("_groundtruth_(1)_trainme_","trainme_original_").lower())