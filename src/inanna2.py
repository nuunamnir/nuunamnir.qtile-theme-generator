import logging
import argparse
import pickle
import colorsys

import numpy
import scipy.spatial.distance
import PIL.Image
import kmedoids

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='{asctime} | {message}', style='{', datefmt='%Y-%m-%dT%H:%M:%S')

    parser = argparse.ArgumentParser(
        prog='inanna',
        description='generates a theme for the qtile window manager',)

    parser.add_argument('-n', dest='name', type=str, help='the name of the theme, if not provided a UUID is generate', required=False, default=None)
    parser.add_argument('-x', '--width', dest='width', type=int, help='target screen resolution width in pixel', required=False, default=3840)
    parser.add_argument('-y', '--height', dest='height', type=int, help='target screen resolution height in pixel', required=False, default=2160)
    parser.add_argument('-w', '--wallpaper', dest='wallpaper', type=str, help='mode in which the image is used as wallpaper in this theme', choices=['stretched', 'centered', 'tiled'], required=False, default='stretched')
    parser.add_argument('-o', '--output', dest='output', type=str, help='output directory path', required=False, default='')
    parser.add_argument('-p', '--preview', dest='preview', type=bool, help='shows a preview of the theme colors', required=False, default=False)
    parser.add_argument(dest='input', type=str, help='an input image based on which the theme is generated')

    args = parser.parse_args()



    image = PIL.Image.open(args.input).convert('RGB')
    pixels = numpy.array(image)
    pixels = pixels.reshape(-1, pixels.shape[2])
    # colors, colors_counts = numpy.unique(pixels, axis=0, return_counts=True)
    # pickle.dump([colors, colors_counts], open('tmp.pkl', 'wb'))
    colors, colors_counts = pickle.load(open('tmp.pkl', 'rb'))
    print(numpy.median(colors_counts), numpy.mean(colors_counts))
    print(len(colors))
    print(len(colors[colors_counts > numpy.mean(colors_counts)]))
    
    d = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(colors[colors_counts > numpy.mean(colors_counts)]))
    result = kmedoids.fasterpam(d, 16)
