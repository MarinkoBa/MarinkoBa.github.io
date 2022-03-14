import os, sys
import PIL
from PIL import Image
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm


def crop_to_multiple_of_n_and_resize(img, n=32, resize_factor=1.0):
    # TODO: crop or rather not maintain aspect ratio exactly and resize? call crop_to... and not resize_to...
    print(f"loaded image of size {img.size}")
    if resize_factor != 1.:
        w, h = img.size
        img = img.resize((int(w*resize_factor), int(h*resize_factor)), resample=PIL.Image.LANCZOS)
        print(f"resized to size {img.size}")
    w, h = img.size
    wr = w - n * (w // n)
    hr = h - n * (h // n)
    if hr != 0 or wr != 0:
        left = wr // 2
        top = hr // 2
        right = w - int(np.ceil(wr/2))
        bottom = h - int(np.ceil(hr/2))
        assert (right - left) % n == 0, f"r, l: {right, left}"
        assert (bottom - top) % n == 0, f"b, t: {bottom, top}"
        img = img.crop((left, top, right, bottom))
    return img




def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i',
                        '--indir',
                        type=str,
                        required=True)

    parser.add_argument('-o',
                        '--outdir',
                        type=str,
                        required=True)

    parser.add_argument('-r',
                        '--resize',
                        default=False,
                        action='store_true',
                        help='Whether or not to resize the image')

    return parser

if __name__ == '__main__':

    parser = get_parser()

    opt = parser.parse_args()

    masks = sorted(glob(os.path.join(opt.indir,  "*_mask.png")))
    images = [x.replace("_mask.png", ".png") for x in masks]
    print(f"Found {len(masks)} inputs.")

    os.makedirs(opt.outdir,exist_ok=True)

    for i,(imageip, maskip) in enumerate(tqdm(zip(images,masks))):
        try:
            image = Image.open(imageip).convert('RGB')
            mask = Image.open(maskip).convert('L')
        except PIL.UnidentifiedImageError as e:
            print('ERROR: ',e)
            continue
        mask_comp = (255 - np.asarray(mask)).astype(bool)
        mask_comp = Image.fromarray(mask_comp)

        masked = Image.composite(image,Image.new('RGB',image.size),mask_comp)

        if opt.resize:
            resize_factor = 1.
            if max(image.size) > 1000:
                resize_factor = 0.6

            image_r = crop_to_multiple_of_n_and_resize(image,resize_factor=resize_factor)
            mask_r = crop_to_multiple_of_n_and_resize(mask,resize_factor=resize_factor)
            masked_r = crop_to_multiple_of_n_and_resize(masked,resize_factor=resize_factor)
        else:
            image_r = image
            mask_r = mask
            masked_r = masked

        maskop = os.path.join(opt.outdir,f'{i}_input_mask.png')
        imgop = os.path.join(opt.outdir, f'{i}_input.png')
        maskedop = os.path.join(opt.outdir, f'{i}.png')

        image_r.save(imgop)
        mask_r.save(maskop)
        masked_r.save(maskedop)





