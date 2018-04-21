import Augmentor
import os

dirpath = os.getcwd()
print(dirpath)

p=Augmentor.Pipeline(dirpath+'\\data\\trainme')
p.ground_truth(dirpath+'\\data\\labelme')


p.rotate(0.7, 25, 25)
p.zoom(0.3, 1.1, 1.6)
p.random_distortion(0.9, 8, 8, 2)

p.sample(10)