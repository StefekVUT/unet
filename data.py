from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import Augmentor


class DataProcess(object):
    """
    Class serves for data manipulation. From conversion to *.npy data format to loading into the neural network.
    """

    def __init__(self, out_rows=512, out_cols=512, data_path='.\\data\\trainme\\output', test_path='.\\data\\test',
                 npy_path='.\\npydata'):
        """
        Initialization of the DataProcess object.
        Setting the image size - all images must have same dimensions.
        Setting the correct folder paths.
        :param out_rows: integer - px size
        :param out_cols: integer - px size
        :param data_path: path to train directory -> images + masks
        :param test_path: path to test directory
        :param npy_path: path for saving *.npy formated data
        """

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = os.path.join(data_path)
        self.test_path = test_path
        self.npy_path = npy_path

    def create_train_data(self, img_type='JPEG'):
        """
        Conversion from native images into *.npy data format for Keras framework
        :param img_type: string Native image type, all images must have same type.
                         Supported img types: [JPEG, jpeg, jpg, png, PNG]
        :return: imgs_train.npy and imgs_mask_train.npy
        """
        i = 0
        print('-'*30)
        print('Converting training images into image and mask files ...')
        print('-'*30)

        img_masks = glob.glob(self.data_path+"\\"+"_groundtruth_*."+img_type)
        img_data = glob.glob(self.data_path+"\\"+"trainme_original_*."+img_type)

        print("Ground truth images : "+str(len(img_masks)))
        print("Train images : "+str(len(img_data)))

        img_data = np.ndarray((len(img_data), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        img_labels = np.ndarray((len(img_masks), self.out_rows, self.out_cols, 1), dtype=np.uint8)

        for midnamedata, midnamelabel in zip(img_data, img_masks):
            img = load_img(midnamedata, grayscale=True)
            label = load_img(midnamelabel, grayscale=True)

            img = img_to_array(img)
            label = img_to_array(label)

            img_data[i] = img
            img_labels[i] = label

            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(img_data)))
            i += 1

        print('loading done')
        np.save(self.npy_path + '\\imgs_train.npy', img_data)
        np.save(self.npy_path + '\\imgs_mask_train.npy', img_labels)
        print('Saving to .npy files done.')

    def create_test_data(self, img_type="JPEG"):
        """
        Conversion from native images into *.npy data format for Keras framework
        :param img_type: string Native image type, all images must have same type.
                         Supported img types: [JPEG, jpeg, jpg, png, PNG]
        :return: imgs_test.npy file
        """
        i = 0
        print('-'*30)
        print('Converting test images ...')
        print('-'*30)
        imgs = glob.glob(self.test_path+"\\"+"*."+img_type)
        print(len(imgs))

        img_data = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for imgname in imgs:
            img = load_img(imgname, grayscale=True)
            img = img_to_array(img)
            img_data[i] = img
            i += 1
        print('loading done')
        np.save(self.npy_path + '\\imgs_test.npy', img_data)
        print('Saving to imgs_test.npy files done.')

    def load_train_data(self, data='imgs_train', mask='imgs_mask_train'):
        """
        function to load npy training and mask data
        :param data: string name of the data file without file type
        :param mask: string name of the mask file without file type
        :return: data and mask variables
        """
        print('-'*30)
        print('loading train images ...')
        print('-'*30)
        imgs_train = np.load(self.npy_path+"\\"+data+".npy")
        imgs_mask_train = np.load(self.npy_path+"\\"+mask+".npy")

        # imgs_train = np.load(self.npy_path + "\\" + data + ".npy")
        # imgs_mask_train = np.load(self.npy_path + "\\" + mask + ".npy")

        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        # mean = imgs_train.mean(axis = 0)
        # imgs_train -= mean
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train, imgs_mask_train

    def load_test_data(self, test='imgs_6_test'):
        """
        function to load npy test image data
        :param test: string name of the data file without filetype
        :return: test image data variable for prediction
        """
        print('-'*30)
        print('load test images...')
        print('-'*30)
        imgs_test = np.load(self.npy_path+"\\"+test+".npy")
        # imgs_test = np.load(self.npy_path + "\\" + test + ".npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        # mean = imgs_test.mean(axis = 0)
        # imgs_test -= mean
        return imgs_test


def generate_images_with_augmentor(train_path='.\\data\\trainme', label_path='.\\data\\labelme', desired_amount=100):
    """
    Call function for augmentor generation tool
    :param train_path: string path to
    :param label_path:
    :param desired_amount:
    :return:
    """
    train = os.path.join(train_path)
    label = os.path.join(label_path)

    p = Augmentor.Pipeline(source_directory=train)
    p.ground_truth(ground_truth_directory=label)

    ####
    # space for adding more functions from Augmentor, varies from personal usecase
    p.rotate(0.7, 25, 25)
    p.zoom(0.3, 1.1, 1.6)
    p.random_distortion(0.9, 8, 8, 2)
    ####
    p.sample(desired_amount)


def save_img(data_npy):
    print("array to image")
    imgs = np.load(data_npy)
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = array_to_img(img)
        img.save("results\\%d.jpg" % i)

if __name__ == "__main__":

    my_data = DataProcess()
    my_data.create_train_data()

    # mydata.create_test_data(img_type='png')
    # imgs_train,imgs_mask_train = mydata.load_train_data()
    # print imgs_train.shape,imgs_mask_train.shape
