import Augmentor

p=Augmentor.Pipeline("C:/Users/smocko/Desktop/generate_data/trainme")
p.ground_truth("C:/Users/smocko/Desktop/generate_data/labelme")
p.set_save_format("auto")

p.rotate(0.7, 25, 25)
p.zoom(0.3, 1.1, 1.6)
p.random_distortion(0.9, 8, 8, 2)

p.sample(10000)