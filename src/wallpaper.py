import argparse
import collections
import itertools
import json
import io
import colorsys

import PIL.Image
import PIL.ImageDraw
import numpy
import scipy.spatial.distance


def create_wallpaper(image_path, width, height):
    im = PIL.Image.open(image_path)
    im_width, im_height = im.size
    background_color = im.getpixel((0, 0))
    wallpaper = PIL.Image.new(mode='RGB', size=(width, height), color=background_color)
    offset = ((width - im_width) // 2, (height - im_height) // 2)
    wallpaper.paste(im, offset)
    return wallpaper


def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v


def visualize_colors(colors, height=64):
    im = PIL.Image.new(mode='RGB', size=(height * len(colors), height))
    canvas = PIL.ImageDraw.Draw(im)
    for i, color in enumerate(colors):
        canvas.rectangle([i * 64, 0, (i + 1) * 64, 64], fill=color)
    return im


def _quantize(image_path):
    im = PIL.Image.open(image_path)
    pixels = numpy.array(im)
    pixels = pixels.reshape(-1, pixels.shape[2])
    
    buckets = []
    buckets.append(pixels)
    while len(buckets) < 16:
        tmp_buckets = []
        for bucket in buckets:
            channel = numpy.argmax(numpy.var(bucket, axis=0))
            threshold = numpy.median(bucket, axis=0)[channel]
            subbucket_a = bucket[bucket[:, channel] <= threshold]
            subbucket_b = bucket[bucket[:, channel] > threshold]
            tmp_buckets.append(subbucket_a)
            tmp_buckets.append(subbucket_b)
        buckets = tmp_buckets

    colors = []
    for i, bucket in enumerate(buckets):
        color = collections.Counter(list(map(tuple, bucket))).most_common(1)
        if not color:
            continue
        colors.append(color[0][0])
    # colors.append((17, 17, 17))
    # colors.append((238, 238, 238))
    highlight_color_idx =numpy.argmax(numpy.sum(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(colors)), axis=1))
    highlight_color = colors[highlight_color_idx]
    del colors[highlight_color_idx]

    colors_by_luminance = sorted(colors, key=lambda x: x[0] * 0.2126 + x[1] * 0.7152 + x[2] * 0.0722)
    # visualize_colors(colors_by_luminance).show()
    foreground_color = colors_by_luminance[-1]
    background_color = colors_by_luminance[0]
    colors_by_saturation = sorted(colors, key=lambda x: rgb_to_hsv(*x)[1])
    # visualize_colors(colors_by_saturation).show()
    
    h, s, v = colorsys.rgb_to_hsv(background_color[0] / 255, background_color[1] / 255, background_color[2] / 255)
    r, g, b = colorsys.hsv_to_rgb(h, s, min(1, v * 2))
    background_accent_color = (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

    h, s, v = colorsys.rgb_to_hsv(foreground_color[0] / 255, foreground_color[1] / 255, foreground_color[2] / 255)
    r, g, b = colorsys.hsv_to_rgb(h, s / 4, v)
    foreground_accent_color = (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


    theme = {
        'foreground': '#%02x%02x%02x' % foreground_color,
        'foreground-accent': '#%02x%02x%02x' % foreground_accent_color,
        'background': '#%02x%02x%02x' % background_color,
        'background-accent': '#%02x%02x%02x' % background_accent_color,
        'highlight': '#%02x%02x%02x' % highlight_color,
    }

    return theme


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='nuunamnir.wallpaper',
        description='generates a wallpaper from an image fitting the current resolution',)
    parser.add_argument('input', help='input image')
    parser.add_argument('output', help='output image')
    parser.add_argument('-x', dest='width', type=int, help='the width of the wallpaper')
    parser.add_argument('-y', dest='height', type=int, help='the height of the wallpaper')
    parser.add_argument('-j', dest='output_json', type=str, help='output json fole')
    
    args = parser.parse_args()

    if args.output_json:
        color_theme = _quantize(args.input)
        with io.open(args.output_json, 'w', encoding='utf-8') as output_handle:
            output_handle.write(json.dumps(color_theme))

    wallpaper = create_wallpaper(args.input, args.width, args.height)
    wallpaper.save(args.output)