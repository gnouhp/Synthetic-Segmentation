import numpy as np
from numpy.random import randint, shuffle
from PIL import Image


animals = ["ostrich", "cheetah", "elephant", "lion"]
animal2size = [[90, 100], [100, 70], [140, 110], [120, 80]]
horiz_shift = [-50, 125, 20, 80]
vert_shift = [10, 40, 80, 140]
sizes = [.8, 1.0, 1.3, 1.8]
colors = [(100, 100, 100), (120, 0, 0), (0, 120, 0), (0, 0, 120)]


def make_sample():
    shuffle(horiz_shift)
    ''' Returns a randomly assembled PIL Image and it's associated semantic mask. '''
    bg = Image.new('RGB', size=(256, 256), color=colors[randint(len(colors))])
    num_animals = randint(3, 5)
    animal_imgs = [(randint(len(animals)), randint(2)) for animal in range(num_animals)]
    category_mask = np.zeros((256, 256))
    
    for idx, animal in enumerate(animal_imgs):
        species = animal[0]
        path2img = "images/{}/{}{}.png".format(animals[species], animals[species], animal[1])
        img = Image.open(path2img)
        img = img.resize((np.array(animal2size[animal[0]]) * sizes[idx]).astype(np.uint8))
        layer_bg = Image.new('RGBA', size=(256, 256), color=(256, 256, 256, 0))
        h_shift = horiz_shift[idx] + randint(-20, 20)
        v_shift = vert_shift[idx] + randint(-20, 20)
        layer_bg.paste(img, (h_shift, v_shift), mask=img)
        layer_np = np.asarray(layer_bg)[:, :, -1]
        pixel_idxes = (layer_np != 0)
        category_mask[pixel_idxes] = species + 1
        bg.paste(layer_bg, (0,0), mask=layer_bg)

    img_np = np.array(bg)
    _min = 0
    _max = len(animals)

    return img_np, category_mask

def generate_dataset(n_samples, imgs_path, masks_path):
    imgs_arr = np.zeros((n_samples, 256, 256, 3))
    masks_arr = np.zeros((n_samples, 256, 256))
    for i in range(n_samples):
        img_np, category_mask = make_sample()
        imgs_arr[i] = img_np
        masks_arr[i] = category_mask
    imgsfp = np.memmap(imgs_path, dtype=np.uint8, mode='w+', shape=imgs_arr.shape)
    masksfp = np.memmap(masks_path, dtype=np.uint8, mode='w+', shape=masks_arr.shape)
    imgsfp[:] = imgs_arr[:]
    masksfp[:] = masks_arr[:]
    del imgsfp  # deleting stores the array values into the disk memory.
    del masksfp
    






