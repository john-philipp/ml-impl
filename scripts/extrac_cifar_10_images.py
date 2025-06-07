import os
import pickle
import numpy as np
from PIL import Image


# CIFAR-10 label names
label_names = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def save_cifar_images(data_dir, output_dir, prefix='train'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all training batches and test batch
    if prefix == 'train':
        batch_files = [f'data_batch_{i}' for i in range(1,6)]
    elif prefix == 'test':
        batch_files = ['test_batch']
    else:
        raise ValueError("prefix must be 'train' or 'test'")

    img_index = 0
    for batch_file in batch_files:
        batch_path = os.path.join(data_dir, batch_file)
        batch = unpickle(batch_path)

        images = batch[b'data']       # shape (10000, 3072), uint8
        labels = batch[b'labels']     # list of ints

        for i, (img_flat, label) in enumerate(zip(images, labels)):
            # Reshape and reorder channels
            r = img_flat[0:1024].reshape(32, 32)
            g = img_flat[1024:2048].reshape(32, 32)
            b = img_flat[2048:3072].reshape(32, 32)

            img = np.stack([r, g, b], axis=2)  # shape (32, 32, 3)

            pil_img = Image.fromarray(img)

            label_name = label_names[label]
            filename = f'{prefix}_{label_name}_{img_index:05d}.jpg'
            pil_img.save(os.path.join(output_dir, filename))

            img_index += 1

    print(f'Saved {img_index} images to {output_dir}')


# Example usage:
data_directory = 'cifar-10-batches-py'  # folder with batch files
train_output_dir = 'cifar10_train_images'
test_output_dir = 'cifar10_test_images'

save_cifar_images(data_directory, train_output_dir, prefix='train')
save_cifar_images(data_directory, test_output_dir, prefix='test')
