
import os
import time
import math

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import *

# code from https://github.com/zhixuhao/unet
from unet.data import trainGenerator
from utils import build_model

def plot_curve(curve_type, train, valid, result_path):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1)
    plt.plot(train, label='Train', color='b')
    plt.plot(valid, label='Validation', color='g')
    plt.legend(loc='upper right', fontsize=12)
    plt.ylabel(curve_type, fontsize=12)
    if curve_type == 'Accuracy':
        plt.ylim([min(plt.ylim()), 1])
    else:
        plt.ylim([0, max(plt.ylim())])
    plt.title('Training and Validation %s' % curve_type)
    plt.xlabel('epoch', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(os.path.join(result_path, '%s.jpg' % curve_type))

def get_train_curve(history, result_path):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plot_curve('Accuracy', acc, val_acc, result_path)
    plot_curve('Loss', loss, val_loss, result_path)

def get_train_data(img_h, img_w, batch_size, data_path, images_folder, labels_folder,
                   valid_images_folder, valid_labels_folder, num_class=2):
    flag_multi_class = True if num_class > 2 else False
    train_gen_args = dict(rotation_range=0.2,
                          width_shift_range=0.05,
                          height_shift_range=0.05,
                          shear_range=0.05,
                          zoom_range=0.05,
                          horizontal_flip=True,
                          vertical_flip = True,
                          fill_mode='nearest')

    train_para = dict(batch_size=batch_size, train_path=data_path, image_color_mode="rgb", mask_color_mode="grayscale",
                       num_class=num_class, flag_multi_class=flag_multi_class,
                       target_size=(img_h, img_w))
    trainGene = trainGenerator(image_folder=images_folder, mask_folder=labels_folder, aug_dict=train_gen_args,
                               **train_para)

    valid_gen_args = dict(fill_mode='nearest')
    validGene = trainGenerator(image_folder=valid_images_folder, mask_folder=valid_labels_folder, aug_dict=valid_gen_args,
                               **train_para)

    return trainGene, validGene

def train_model(img_w, img_h, data_path, epochs, batch_size, cache_path, lr=1e-4, num_class=2, model_path=None, use_unet=False):
    model = build_model((img_h, img_w, 3), num_class=num_class, model_path=model_path, use_unet=use_unet)

    loss_type = 'categorical_crossentropy' if num_class > 2 else 'binary_crossentropy'
    model.compile(optimizer=Adam(lr=lr), loss=loss_type, metrics=['accuracy'])

    now = time.strftime("%y-%m-%d_%H_%M", time.localtime(time.time()))
    prefix = 'U_' if use_unet else 'X_'
    model_name = prefix + "_" + now + '_{epoch:02d}-{val_acc:.3f}.hdf5'
    abs_model_name = os.path.join(cache_path, model_name)

    model_checkpoint = ModelCheckpoint(abs_model_name, monitor='val_loss', verbose=2, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=7)
    callbacks = [early_stop, model_checkpoint]

    images_folder = 'images'
    labels_folder = 'labels'
    valid_images_folder = 'valid_images'
    valid_labels_folder = 'valid_labels'
    train_samples = len(os.listdir(os.path.join(data_path, images_folder)))
    valid_samples = len(os.listdir(os.path.join(data_path, valid_images_folder)))

    trainGene, validGene = get_train_data(img_h, img_w, batch_size, data_path, images_folder, labels_folder,
                                            valid_images_folder, valid_labels_folder, num_class)
    history = model.fit_generator(trainGene, steps_per_epoch=math.ceil(train_samples / batch_size), epochs=epochs,
                        validation_data=validGene, validation_steps=math.ceil(valid_samples / batch_size),
                        callbacks=callbacks)

    get_train_curve(history, data_path)


if __name__ == '__main__':
    img_w, img_h = (512, 512)
    data_path = r'F:\Biliopancreatic-EUS-IREAD\anatomical_structures_localization'
    cache_path = os.path.join(data_path, 'cache')
    
    epochs = 100
    batch_size = 2
    lr = 1e-4
    num_class = 2
    model_path = None
    use_unet = False
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    train_model(img_w, img_h, data_path, epochs, batch_size, cache_path, lr, num_class, model_path, use_unet)
