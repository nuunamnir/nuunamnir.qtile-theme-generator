#!/usr/bin/env python
'''
TODO:
~~~1. debug mode~~~
~~~2. extract colors from image~~~
3. check for legability rules and apply corrections
~~~4. fix resizing of wallpaper~~~
~~~5. add argparser~~~
~~~6. update README.md~~~
'''
import os
import re
import io
import logging
import json
import argparse

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

import matplotlib
import matplotlib.pyplot

import numpy
import scipy.cluster.vq
import colorspacious

theme = {
    'colors': {
        'background': '#ffffff',
        'foreground': '#000000',
        'foreground-accent': '#ff0000',
        'foreground-accent-alt1': '#00ff00',
        'foreground-accent-alt2': '#0000ff',
        'foreground-accent-complementary': '#00ffff',
        'background-accent': '#003333',
    },
    'fonts': {
        'standard': 'FiraCode Nerd Font',
    }
}

def generate_debug_preview(theme=None):
    image = PIL.Image.new(size=(512, 512), mode='RGB', color=(34, 34, 34))
    canvas = PIL.ImageDraw.Draw(image)
    canvas.rounded_rectangle((8, 8, 512 - 8, 512 - 8), fill=theme['colors']['background'], radius=8)
    font = matplotlib.font_manager.FontProperties(family=theme['fonts']['standard'], weight='bold')
    font_file = matplotlib.font_manager.findfont(font)
    standard_font = PIL.ImageFont.truetype(font_file, 12)
    canvas.text((16, 16), 'Lorem ipsum dolor sit amet, consetetur sadipscing elitr,\nsed diam nonumy eirmod tempor ...', fill=theme['colors']['foreground'], font=standard_font)
    canvas.rounded_rectangle((12, 60, 512-12, 60 + 40), fill=theme['colors']['background-accent'], radius=8)
    canvas.text((16, 64), '... invidunt ut labore et dolore magna aliquyam erat,\nsed diam voluptua.', fill=theme['colors']['foreground-accent'], font=standard_font)
    
    canvas.rounded_rectangle((12, 60 + 60, 512-12, 60 + 60 + 40), fill=theme['colors']['background-accent'], radius=8)
    canvas.text((16, 64 + 60), 'At vero eos et accusam et justo duo dolores et ea rebum.', fill=theme['colors']['foreground-accent-alt1'], font=standard_font)
    
    canvas.rounded_rectangle((12, 60 + 120, 512-12, 60 + 120 + 40), fill=theme['colors']['background-accent'], radius=8)
    canvas.text((16, 64 + 120), 'Stet clita kasd gubergren, no sea takimata sanctus est.', fill=theme['colors']['foreground-accent-alt2'], font=standard_font)
    
    canvas.rounded_rectangle((12, 60 + 180, 512-12, 60 + 180 + 40), fill=theme['colors']['background-accent'], radius=8)
    canvas.text((16, 64 + 180), 'Ut enim ad minim veniam, quis nostrud exercitation ullamco\nlaboris nisi ut aliquid ex ea commodi consequat.', fill=theme['colors']['foreground-accent-complementary'], font=standard_font)
    
    canvas.rounded_rectangle((12, 60 + 240, 512-12, 60 + 240 + 40), fill=theme['colors']['foreground-accent-alt1'], radius=8)
    canvas.text((16, 64 + 240), 'Nam liber tempor cum soluta nobis eleifend option congue nihil\nimperdiet doming id quod mazim placerat facer possim assum.', fill=theme['colors']['foreground-accent-complementary'], font=standard_font)


    canvas.rounded_rectangle((12, 460, 44, 492), fill=theme['colors']['background'], outline=theme['colors']['foreground'], radius=8)
    canvas.text((12 + 4, 460 + 4, 44, 492), 'bg', fill=theme['colors']['foreground'], font=standard_font)

    canvas.rounded_rectangle((12 + 34, 460, 44 + 34, 492), fill=theme['colors']['background-accent'], outline=theme['colors']['foreground-accent'], radius=8)
    canvas.text((12 + 34+ 4, 460+ 4, 44 + 34, 492), 'ba', fill=theme['colors']['foreground-accent'], font=standard_font)

    canvas.rounded_rectangle((12 + 2 * 34, 460, 44 + 2 * 34, 492), fill=theme['colors']['foreground-accent-complementary'], outline=theme['colors']['background-accent'], radius=8)
    canvas.text((12 + 2 * 34+ 4, 460+ 4, 44 + 2 * 34, 492), 'bac', fill=theme['colors']['background-accent'], font=standard_font)

    canvas.rounded_rectangle((12 + 3 * 34, 460, 44 + 3 * 34, 492), fill=theme['colors']['foreground-accent-alt2'], outline=theme['colors']['background-accent'], radius=8)
    canvas.text((12 + 3 * 34+ 4, 460+ 4, 44 + 3 * 34, 492), 'fa2', fill=theme['colors']['background-accent'], font=standard_font)

    canvas.rounded_rectangle((12 + 4 * 34, 460, 44 + 4 * 34, 492), fill=theme['colors']['foreground-accent-alt1'], outline=theme['colors']['background-accent'], radius=8)
    canvas.text((12 + 4 * 34+ 4, 460+ 4, 44 + 4 * 34, 492), 'fa1', fill=theme['colors']['background-accent'], font=standard_font)

    canvas.rounded_rectangle((12 + 5 * 34, 460, 44 + 5 * 34, 492), fill=theme['colors']['foreground-accent'], outline=theme['colors']['background-accent'], radius=8)
    canvas.text((12 + 5 * 34+ 4, 460+ 4, 44 + 5 * 34, 492), 'fa', fill=theme['colors']['background-accent'], font=standard_font)

    canvas.rounded_rectangle((12 + 6 * 34, 460, 44 + 6 * 34, 492), fill=theme['colors']['foreground'], outline=theme['colors']['background'], radius=8)
    canvas.text((12 + 6 * 34+ 4, 460+ 4, 44 + 6 * 34, 492), 'fg', fill=theme['colors']['background'], font=standard_font)

    canvas.rounded_rectangle((12 + 34, 460 - 36, 44 + 34, 492 - 36), fill=theme['colors']['color1'], outline=theme['colors']['foreground-accent'], radius=8)
    canvas.rounded_rectangle((12 + 34 * 2, 460 - 36, 44 + 34* 2, 492 - 36), fill=theme['colors']['color2'], outline=theme['colors']['foreground-accent'], radius=8)
    canvas.rounded_rectangle((12 + 34 * 3, 460 - 36, 44 + 34* 3, 492 - 36), fill=theme['colors']['color3'], outline=theme['colors']['foreground-accent'], radius=8)
    canvas.rounded_rectangle((12 + 34 * 4, 460 - 36, 44 + 34* 4, 492 - 36), fill=theme['colors']['color4'], outline=theme['colors']['foreground-accent'], radius=8)
    canvas.rounded_rectangle((12 + 34 * 5, 460 - 36, 44 + 34* 5, 492 - 36), fill=theme['colors']['color5'], outline=theme['colors']['foreground-accent'], radius=8)
    canvas.rounded_rectangle((12 + 34 * 6, 460 - 36, 44 + 34* 6, 492 - 36), fill=theme['colors']['color6'], outline=theme['colors']['foreground-accent'], radius=8)
    canvas.rounded_rectangle((12 + 34 * 7, 460 - 36, 44 + 34* 7, 492 - 36), fill=theme['colors']['color7'], outline=theme['colors']['foreground-accent'], radius=8)
    canvas.rounded_rectangle((12 + 34 * 8, 460 - 36, 44 + 34* 8, 492 - 36), fill=theme['colors']['color8'], outline=theme['colors']['foreground-accent'], radius=8)

    canvas.rounded_rectangle((12 + 5 * 34, 460 - 72, 44 + 5 * 34, 492 - 72), fill=theme['colors']['color9'], outline=theme['colors']['background-accent'], radius=8)
    canvas.rounded_rectangle((12 + 6 * 34, 460 - 72, 44 + 6 * 34, 492 - 72), fill=theme['colors']['color10'], outline=theme['colors']['background-accent'], radius=8)
    canvas.rounded_rectangle((12 + 7 * 34, 460 - 72, 44 + 7 * 34, 492 - 72), fill=theme['colors']['color11'], outline=theme['colors']['background-accent'], radius=8)
    canvas.rounded_rectangle((12 + 8 * 34, 460 - 72, 44 + 8 * 34, 492 - 72), fill=theme['colors']['color12'], outline=theme['colors']['background-accent'], radius=8)
    canvas.rounded_rectangle((12 + 9 * 34, 460 - 72, 44 + 9 * 34, 492 - 72), fill=theme['colors']['color13'], outline=theme['colors']['background-accent'], radius=8)
    canvas.rounded_rectangle((12 + 10 * 34, 460 - 72, 44 + 10 * 34, 492 - 72), fill=theme['colors']['color14'], outline=theme['colors']['background-accent'], radius=8)
    canvas.rounded_rectangle((12 + 11 * 34, 460 - 72, 44 + 11 * 34, 492 - 72), fill=theme['colors']['color15'], outline=theme['colors']['background-accent'], radius=8)
    canvas.rounded_rectangle((12 + 12 * 34, 460 - 72, 44 + 12 * 34, 492 - 72), fill=theme['colors']['color16'], outline=theme['colors']['background-accent'], radius=8)
    
    image.save('debug.png')

    matplotlib.pyplot.imshow(image)
    matplotlib.pyplot.show()


def extract_colors_from_image(image_file_path, wallpaper_width, wallpaper_height, wallpaper_mode='stretched', light=False, debug=False):
    image = PIL.Image.open(image_file_path).convert('RGB')

    image_resized = image.resize((128, 128), resample=PIL.Image.LANCZOS)
    pixels = numpy.array(image_resized)
    pixels = pixels.reshape(-1, pixels.shape[2]) / 255.

    retries = 0
    while retries < 128:
        try:
            colors, _ = scipy.cluster.vq.kmeans2(pixels, 16, missing='raise')
            break
        except scipy.cluster.vq.ClusterError:
            retries += 1
            continue
    if retries == 128:
        logging.error(f'𝍐 | the provided pictures does not contain sufficient colors to create a theme')
        raise ValueError

    colors_srgb = [[f'#{r:x}{g:x}{b:x}', (r, g, b)] for r, g, b in numpy.round(colorspacious.cspace_convert(colors, 'sRGB1', 'sRGB255')).astype(int) if re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', f'#{r:x}{g:x}{b:x}')]

    colors_lch = list()
    for color_str, color_tup in colors_srgb:
        colors_lch.append(colorspacious.cspace_convert(color_tup, 'sRGB255', 'CIELCh'))


    if light:
       colors_lch = sorted(colors_lch, key=lambda x: -x[0])
    else:
        colors_lch = sorted(colors_lch, key=lambda x: x[0])

    colors_lch[-4][2] = (colors_lch[-3][2] + 90) % 360
    colors_lch[-5] = colors_lch[-2]
    colors_lch[-5][2] = (colors_lch[-2][2] + 180) % 360
    if colors_lch[-3][0] - colors_lch[-5][0] < 50:
        colors_lch[-5][0] = colors_lch[-3][0] + 50
        logging.warning('𝍄 | contrast between foreground-accent-complementary and foreground-accent-alt1 too low - darkening')
    if colors_lch[1][0] - colors_lch[-5][0] < 50:
        colors_lch[-5][0] = colors_lch[1][0] + 50
        logging.warning('𝍄 | contrast between foreground-accent-complementary and background-accent too low - darkening')
    if abs(colors_lch[-3][2] - colors_lch[-5][2]) % 360 < 10:
        colors_lch[-5][2] = abs(colors_lch[-3][2] + 10) % 360
        logging.warning('𝍄 | hue between foreground-accent-complementary and background-accent too similar - rotating')

    palette = [None] * 16
    for i in range(8):
        palette[i] = colors_lch[1].copy()
        palette[i + 8] = colors_lch[-2].copy()

    palette[0][2] += 0
    palette[1][2] += 45
    palette[2][2] += 90
    palette[3][2] += 135
    palette[4][2] += 180
    palette[5][2] += 215
    palette[6][2] += 260
    palette[7][2] += 305

    palette[8][2] += 0
    palette[9][2] += 45
    palette[10][2] += 90
    palette[11][2] += 135
    palette[12][2] += 180
    palette[13][2] += 215
    palette[14][2] += 260
    palette[15][2] += 305

    palette_colors = numpy.round(colorspacious.cspace_convert(palette, 'CIELCh', 'sRGB255')).astype(int)
    palette_srgb = [[f'#{min(abs(r), 255):{0}2x}{min(abs(g), 255):{0}2x}{min(abs(b), 255):{0}2x}', (min(abs(r), 255), min(abs(g), 255), min(abs(b), 255))] for r, g, b in palette_colors] # if re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', f'#{r:x}{g:x}{b:x}')]

    colors = numpy.round(colorspacious.cspace_convert(colors_lch, 'CIELCh', 'sRGB255')).astype(int)
    colors_srgb = [[f'#{min(abs(r), 255):{0}2x}{min(abs(g), 255):{0}2x}{min(abs(b), 255):{0}2x}', (min(abs(r), 255), min(abs(g), 255), min(abs(b), 255))] for r, g, b in colors] # if re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', f'#{r:x}{g:x}{b:x}')]

    if debug:
        matplotlib.pyplot.imshow(image)
        matplotlib.pyplot.show()

    wallpaper = PIL.Image.new(size=(wallpaper_width, wallpaper_height), mode='RGB', color=colors_srgb[0][0])    
    wallpaper_image = image.copy()
    wallpaper_image.thumbnail((wallpaper_width, wallpaper_height), PIL.Image.Resampling.LANCZOS)
    wallpaper_image_width, wallpaper_image_height = wallpaper_image.size
    offset = ((wallpaper_width - wallpaper_image_width) // 2, (wallpaper_height - wallpaper_image_height) // 2)
    wallpaper.paste(wallpaper_image, offset)
    if wallpaper_mode not in ['stretched', 'centered', 'tiled']:
        logging.error(f'𝍐 | the provided wallpaper mode is not supported, only streteched, centered and tiled are supported')
        raise ValueError

    return wallpaper, {
        'colors': {
            'background': colors_srgb[0][0],
            'foreground': colors_srgb[-1][0],
            'foreground-accent': colors_srgb[-2][0],
            'foreground-accent-alt1': colors_srgb[-3][0],
            'foreground-accent-alt2': colors_srgb[-4][0],
            'foreground-accent-complementary': colors_srgb[-5][0],
            'background-accent': colors_srgb[1][0],
            'color1': palette_srgb[0][0],
            'color2': palette_srgb[1][0],
            'color3': palette_srgb[2][0],
            'color4': palette_srgb[3][0],
            'color5': palette_srgb[4][0],
            'color6': palette_srgb[5][0],
            'color7': palette_srgb[6][0],
            'color8': palette_srgb[7][0],
            'color9': palette_srgb[8][0],
            'color10': palette_srgb[9][0],
            'color11': palette_srgb[10][0],
            'color12': palette_srgb[11][0],
            'color13': palette_srgb[12][0],
            'color14': palette_srgb[13][0],
            'color15': palette_srgb[14][0],
            'color16': palette_srgb[15][0],
        },
        'fonts': {
            'standard': 'FiraCode Nerd Font',
        },
        'wallpaper': wallpaper_mode,
}


def save(name, theme, wallpaper, output_dir_path=''):
    try:
        theme_dir_path = os.path.join(output_dir_path, name)
        os.makedirs(theme_dir_path)
        with io.open(os.path.join(theme_dir_path, 'theme.json'), 'w', encoding='utf-8') as output_handle:
            json.dump(theme, output_handle)
        wallpaper.save(os.path.join(theme_dir_path, 'wallpaper.png'))
    except FileExistsError:
        logging.error(f'𝍐 | a theme of this name {name} already exists')



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='{asctime} | {message}', style='{', datefmt='%Y-%m-%dT%H:%M:%S')

    parser = argparse.ArgumentParser(
        prog='inanna',
        description='generates a theme for the qtile window manager',)

    parser.add_argument('-n', dest='name', type=str, help='the name of the theme, if not provided a UUID is generate', required=False, default=None)
    parser.add_argument('-x', '--width', dest='width', type=int, help='target screen resolution width in pixel', required=False, default=3840)
    parser.add_argument('-y', '--height', dest='height', type=int, help='target screen resolution height in pixel', required=False, default=2160)
    parser.add_argument('-w', '--wallpaper', dest='wallpaper_mode', type=str, help='mode in which the image is used as wallpaper in this theme', choices=['stretched', 'centered', 'tiled'], required=False, default='stretched')
    parser.add_argument('-o', '--output', dest='output', type=str, help='output directory path', required=False, default='')
    parser.add_argument('-p', '--preview', dest='preview', help='shows a preview of the theme colors', required=False, default=False, action='store_true')
    parser.add_argument('-l', '--light', dest='light', help='flips colors to create a light theme', required=False, default=False, action='store_true')
    parser.add_argument('-d', '--debug', dest='debug', help='toggles between debug and production mode', required=False, default=False, action='store_true')
    parser.add_argument('-fs', '--font-standard', type=str, dest='font_standard', help='standard font for the theme', required=False, default='FiraCode Nerd Font')
    parser.add_argument('-fc', '--font-console', type=str, dest='font_console', help='console (monospace) font for the theme', required=False, default='Fira Code Regular')
    parser.add_argument('-ff', '--font-features', type=str, dest='font_features', help='font features for the theme', required=False, default='FiraCode-Regular +cv16 +ss02')
    
    
    parser.add_argument(dest='input', type=str, help='an input image based on which the theme is generated')
   
    args = parser.parse_args()

    if args.debug:
        logger = logging.getLogger()
        logger.setLevel(logging.ERROR)
   
    image_file_path = args.input
    wallpaper, theme = extract_colors_from_image(image_file_path, args.width, args.height, wallpaper_mode=args.wallpaper_mode, light=args.light, debug=args.debug)
    if args.debug or args.preview:
        generate_debug_preview(theme)
    theme['fonts'] = {
        'standard': args.font_standard,
        'console': args.font_console,
        'features': args.font_features,
    }
    # save(args.name, theme, wallpaper, args.output)