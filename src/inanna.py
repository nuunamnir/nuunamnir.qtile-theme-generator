#!/usr/bin/env python
# chmod +x inanna.py
import os
import io
import argparse
import uuid
import logging
import json
import colorsys

import PIL.Image
import PIL.ImageDraw
import numpy
import scipy.optimize


class Theme:
    def __init__(self, input_path, width=3840, height=2160, wallpaper_mode='stretched', name=None, light=False):
        self.image = PIL.Image.open(input_path).convert('RGB')

        if name is None:
            self.name = uuid.uuid4().hex
        else:
            self.name = name

        self.theme_data = dict()
        self.theme_data['name'] = self.name
        self.theme_data['colors'] = dict()
        self.theme_data['fonts'] = dict()
        self.light = light

        self.theme_data['wallpaper'] = wallpaper_mode

        self._quantize()

        if wallpaper_mode == 'stretched' or wallpaper_mode == 'tiled':
            self.wallpaper = self.image
        elif wallpaper_mode == 'centered':
            self.wallpaper = PIL.Image.new(mode='RGB', size=(width, height), color=self.theme_data['colors']['background-accent'])
            offset = ((width - im_width) // 2, (height - im_height) // 2)
            self.wallpaper.paste(im, offset)
        else:
            raise ValueError


    def _quantize(self, n=16):
        pixels = numpy.array(self.image)
        pixels = pixels.reshape(-1, pixels.shape[2]) / 255.

        partitions = list()
        partitions.append(pixels)
        # TODO: handle images that do not contain at leat 16 unique colors
        while len(partitions) < n:
            tmp_partitions = list()
            for partition in partitions:
                channel = numpy.argmax(numpy.var(partition, axis=0))
                threshold = numpy.median(partition, axis=0)[channel]
                sub_partition_left = partition[partition[:, channel] <= threshold]
                sub_partition_right = partition[partition[:, channel] > threshold]
                if len(sub_partition_left) > 0:
                    tmp_partitions.append(sub_partition_left)
                if len(sub_partition_right) > 0:
                    tmp_partitions.append(sub_partition_right)
            if len(tmp_partitions) > 0:
                partitions = tmp_partitions
            else:
                break

        self.colors = list()
        for partition in partitions:
            if len(partition) > 0:
                self.colors.append(numpy.mean(partition, axis=0))
            # values, counts = numpy.unique(partition, axis=1, return_counts=True)
            # if len(values) > 0:
            #     self.colors.append(values[numpy.argmax(counts)])

        
        # whitened_pixels = scipy.cluster.vq.whiten(pixels)
        # self.colors, _ = scipy.cluster.vq.kmeans(whitened_pixels, n)
        # self.colors *= 255
        # self.colors = numpy.round(self.colors).astype(int)

        # self.colors, _ = scipy.cluster.vq.kmeans2(pixels.astype(float), n, minit='++', iter=512)
        # self.colors = numpy.round(self.colors).astype(int)

        # luminance = lambda x: x[0] * 0.2126 + x[1] * 0.7152 + x[2] * 0.0722
        luminance = lambda x: colorsys.rgb_to_hls(*x)[1]
        colors_by_luminance = sorted(self.colors, key=lambda x: luminance(x))



        # adapted_luminance = lambda x, target_luminance :numpy.abs(target_luminance - (x[0] * 0.2126 + x[1] * 0.7152 + x[2] * 0.0722))
        adapted_luminance = lambda x, target_luminance :numpy.abs(target_luminance - colorsys.rgb_to_hls(*x)[1])

        luminance_foreground = luminance(colors_by_luminance[-1])
        luminance_background = luminance(colors_by_luminance[0])

        method='L-BFGS-B'

        if luminance_foreground - luminance_background < 0.5:
            logging.warning('𝍄 | luminance stretch needs to be applied')
            if luminance_background > 0.5:
                logging.warning('𝍄 | background is too bright')
                colors_by_luminance[0] = scipy.optimize.minimize(adapted_luminance, x0=colors_by_luminance[0], bounds=((0, 1), (0, 1), (0, 1)), args=(0.5), method=method).x
                colors_by_luminance[-1] = numpy.array([1, 1, 1])
            else:
                logging.warning('𝍄 | adjusting foreground')
                colors_by_luminance[-1] = scipy.optimize.minimize(adapted_luminance, x0=colors_by_luminance[-1], bounds=((0,1), (0, 1), (0, 1)), args=(luminance_background + 0.5), method=method).x
        
        colors_by_saturation = numpy.array(sorted(colors_by_luminance[1:-1], key=lambda x: colorsys.rgb_to_hls(*x)[2]))
        distances = scipy.spatial.distance.cdist([colors_by_luminance[0]], colors_by_saturation)
        colors_by_saturation = colors_by_saturation[distances.flatten() > 0.125]
        distances = scipy.spatial.distance.cdist([colors_by_luminance[-1]], colors_by_saturation)
        colors_by_saturation = colors_by_saturation[distances.flatten() > 0.125]
        accent_color = colors_by_saturation[-1]

        if luminance_foreground - luminance(accent_color) > 0.125:
            logging.warning('𝍄 | adjusting foreground accent - darkening')
            foreground_accent  = scipy.optimize.minimize(adapted_luminance, x0=accent_color, bounds=((0, 1), (0, 1), (0, 1)), args=(luminance_foreground - 0.125), method=method).x
        elif luminance_foreground - luminance(accent_color) < 0.0625:
            logging.warning('𝍄 | adjusting foreground accent - brightening')
            foreground_accent  = scipy.optimize.minimize(adapted_luminance, x0=accent_color, bounds=((0, 1), (0, 1), (0, 1)), args=(luminance_foreground - 0.0625), method=method).x
        else:
            foreground_accent = accent_color

        if luminance(accent_color) - luminance_background > 0.125:
            logging.warning('𝍄 | adjusting background accent - darkening')
            background_accent  = scipy.optimize.minimize(adapted_luminance, x0=accent_color, bounds=((0, 1), (0, 1), (0, 1)), args=(luminance_background + 0.125), method=method).x
        elif luminance(accent_color) - luminance_background < 0.0625:
            logging.warning('𝍄 | adjusting background accent - brightening')
            background_accent  = scipy.optimize.minimize(adapted_luminance, x0=accent_color, bounds=((0, 1), (0, 1), (0, 1)), args=(luminance_background + 0.0625), method=method).x
        else:
            background_accent = accent_color


        adjacency = 0.05
        if not self.light:
            h, s, v = colorsys.rgb_to_hsv(*foreground_accent)
            h_alt1 = h + adjacency if h <= 1 - adjacency else 1 - (h + adjacency)
            h_alt2 = h - adjacency if h >= adjacency else 1 + (h - adjacency)
            alt1 = numpy.array(colorsys.hsv_to_rgb(h_alt1, s, v))
            alt2 = numpy.array(colorsys.hsv_to_rgb(h_alt2, s, v))

            h_complimentary = h - 0.5 if h >= 0.5 else 1 + (h - 0.5)
            complimentary = numpy.array(colorsys.hsv_to_rgb(h_complimentary, s, v))


            self.theme_data['colors']['foreground'] = '#%02x%02x%02x' % tuple(numpy.floor(colors_by_luminance[-1] * 255).astype(int))
            self.theme_data['colors']['foreground-accent'] = '#%02x%02x%02x' % tuple(numpy.floor(foreground_accent * 255).astype(int))
            self.theme_data['colors']['foreground-accent-alt1'] = '#%02x%02x%02x' % tuple(numpy.floor(alt1 * 255).astype(int))
            self.theme_data['colors']['foreground-accent-alt2'] = '#%02x%02x%02x' % tuple(numpy.floor(alt2 * 255).astype(int))
            self.theme_data['colors']['foreground-accent-complimentary'] = '#%02x%02x%02x' % tuple(numpy.floor(complimentary * 255).astype(int))
            self.theme_data['colors']['background-accent'] = '#%02x%02x%02x' % tuple(numpy.floor(background_accent * 255).astype(int))
            self.theme_data['colors']['background'] = '#%02x%02x%02x' % tuple(numpy.floor(colors_by_luminance[0] * 255).astype(int))
        else:
            h, s, v = colorsys.rgb_to_hsv(*background_accent)
            h_alt1 = h + adjacency if h <= 1 - adjacency else 1 - (h + adjacency)
            h_alt2 = h - adjacency if h >= adjacency else 1 + (h - adjacency)
            alt1 = numpy.array(colorsys.hsv_to_rgb(h_alt1, s, v))
            alt2 = numpy.array(colorsys.hsv_to_rgb(h_alt2, s, v))

            h_complimentary = h - 0.5 if h >= 0.5 else 1 + (h - 0.5)
            complimentary = numpy.array(colorsys.hsv_to_rgb(h_complimentary, s, v))

            self.theme_data['colors']['foreground'] = '#%02x%02x%02x' % tuple(numpy.floor(colors_by_luminance[0] * 255).astype(int))
            self.theme_data['colors']['foreground-accent'] = '#%02x%02x%02x' % tuple(numpy.floor(background_accent * 255).astype(int))
            self.theme_data['colors']['foreground-accent-alt1'] = '#%02x%02x%02x' % tuple(numpy.floor(alt1 * 255).astype(int))
            self.theme_data['colors']['foreground-accent-alt2'] = '#%02x%02x%02x' % tuple(numpy.floor(alt2 * 255).astype(int))
            self.theme_data['colors']['foreground-accent-complimentary'] = '#%02x%02x%02x' % tuple(numpy.floor(complimentary * 255).astype(int))
            self.theme_data['colors']['background-accent'] = '#%02x%02x%02x' % tuple(numpy.floor(foreground_accent * 255).astype(int))
            self.theme_data['colors']['background'] = '#%02x%02x%02x' % tuple(numpy.floor(colors_by_luminance[-1] * 255).astype(int))



    def preview(self, size=64, padding=4):
        preview_width = len(self.theme_data['colors'].keys()) * (size + padding) + padding
        preview_height = size + 3 * padding + self.image.height * ((preview_width - 2 * padding) / self.image.width)
        preview_wallpaper = self.image.resize(size=tuple(numpy.round((preview_width - 2 * padding, self.image.height * ((preview_width - 2 * padding) / self.image.width))).astype(int)))
        preview = PIL.Image.new(size=tuple(numpy.round((preview_width, preview_height)).astype(int)), mode='RGB')
        canvas = PIL.ImageDraw.Draw(preview)
        for i, color in enumerate([self.theme_data['colors']['foreground'], self.theme_data['colors']['foreground-accent-alt1'], self.theme_data['colors']['foreground-accent'], self.theme_data['colors']['foreground-accent-alt2'], self.theme_data['colors']['foreground-accent-complimentary'], self.theme_data['colors']['background-accent'], self.theme_data['colors']['background']]):
            canvas.rectangle((i * (size + padding) + padding, padding, (i + 1) * (size + padding), padding + size), fill=color)

        preview.paste(preview_wallpaper, (padding, 2 * padding + size))
        preview.show()


    def save(self, output_dir_path=''):
        try:
            theme_dir_path = os.path.join(output_dir_path, self.name)
            os.makedirs(theme_dir_path)
            with io.open(os.path.join(theme_dir_path, 'theme.json'), 'w', encoding='utf-8') as output_handle:
                json.dump(self.theme_data, output_handle)
            self.wallpaper.save(os.path.join(theme_dir_path, 'wallpaper.png'))
        except FileExistsError:
            logging.error(f'𝍐 | a theme of this name {self.name} already exists')


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
    parser.add_argument('-l', '--light', dest='light', type=bool, help='flips colors to create a light theme', required=False, default=False)
    parser.add_argument(dest='input', type=str, help='an input image based on which the theme is generated')

    args = parser.parse_args()

    theme = Theme(args.input, name=args.name, width=args.width, height=args.height, wallpaper_mode=args.wallpaper, light=args.light)
    if args.preview:
        theme.preview()
    theme.save(output_dir_path=args.output)