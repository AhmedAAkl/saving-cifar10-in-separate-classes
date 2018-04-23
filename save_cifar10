"""
@author: A.Akl

this code snippet is used for saving cifar10 binary dataset classes in separate folders
All you need is data_path and create data_output path including train and test
folders and let the code do the rest for you.

"""


import numpy as np
import pickle
import os
import scipy.misc

# binary dataset files path
data_path = '/cifar-10-batches-py/'

# output directories
output_dir = '/cifar10/'

img_size = 32

num_channels = 3

num_files_train = 5

images_per_file = 10000

num_images_train = num_files_train * images_per_file


def load_data(filename):

     # Create full path for the file.
    file_path = os.path.join(data_path,filename)
    
    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:        
        data = pickle.load(file, encoding='bytes')

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])
    
    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw_images, dtype=float) / 255.0
    
    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    return images, cls

# separate methods for different classes
def save_class0(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class0')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)



def save_class1(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class1')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)
        



def save_class2(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class2')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)
        


def save_class3(save_dir, image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class3')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)
        



def save_class4(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class4')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)
        


def save_class5(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class5')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)
        


def save_class6(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class6')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)
        


def save_class7(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class7')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)
        


def save_class8(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class8')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)
        


def save_class9(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class9')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)
        

# dict instead of switch case or if else technique
class_label = {
        0: save_class0,
        1: save_class1,
        2: save_class2,
        3: save_class3,
        4: save_class4,
        5: save_class5,
        6: save_class6,
        7: save_class7,
        8: save_class8,
        9: save_class9
        }
    
# saving training data
i = 0
for num in range(1,num_files_train + 1):
        
    images,labels = load_data('data_batch_' + str(num))
    
    for image,label in zip(images,labels):
        image = image.transpose([1,2,0])
        class_label[label]('train',image,i) # call dict as method
        i += 1
        



# saving test data
images,labels = load_data('test_batch')
i = 0
for image,label in zip(images,labels):
    image = image.transpose([1,2,0])
    class_label[label]('test',image,i)
    i += 1
