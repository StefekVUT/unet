#from unet import *
from unet2class import *
from data import *

mydata = dataProcess(512,512)

imgs_test = mydata.load_test_data()

#myunet = myUnet()
myunet = MyUnet()

model = myunet.get_unet()

model.load_weights('unet.hdf5')

imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

np.save('imgs_mask_test.npy', imgs_mask_test)

myunet.save_img()