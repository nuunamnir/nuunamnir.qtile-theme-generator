'''
TODO:
1. debug mode
2. extract colors from image
3. check for legability rules and apply corrections
'''
import os
import re

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
    font = matplotlib.font_manager.FontProperties(family=theme['fonts']['standard'], weight='regular')
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
    
    canvas.rounded_rectangle((12, 460, 44, 492), fill=theme['colors']['background'], outline=theme['colors']['foreground'], radius=8)
    canvas.rounded_rectangle((12 + 34, 460, 44 + 34, 492), fill=theme['colors']['background-accent'], outline=theme['colors']['foreground-accent'], radius=8)
    canvas.rounded_rectangle((12 + 2*34, 460, 44+ 2*34, 492), fill=theme['colors']['foreground-accent-complementary'], outline=theme['colors']['background-accent'], radius=8)
    canvas.rounded_rectangle((12+ 3*34, 460, 44+ 3*34, 492), fill=theme['colors']['foreground-accent-alt2'], outline=theme['colors']['background-accent'], radius=8)
    canvas.rounded_rectangle((12+ 4*34, 460, 44+ 4*34, 492), fill=theme['colors']['foreground-accent-alt1'], outline=theme['colors']['background-accent'], radius=8)
    canvas.rounded_rectangle((12+ 5*34, 460, 44+ 5*34, 492), fill=theme['colors']['foreground-accent'], outline=theme['colors']['background-accent'], radius=8)
    canvas.rounded_rectangle((12+ 6*34, 460, 44+ 6*34, 492), fill=theme['colors']['foreground'], outline=theme['colors']['background'], radius=8)


    matplotlib.pyplot.imshow(image)
    matplotlib.pyplot.show()


def extract_colors_from_image(image_file_path):
    image = PIL.Image.open(image_file_path).convert('RGB')
    image = image.resize((128, 128), resample=PIL.Image.LANCZOS)
    pixels = numpy.array(image)
    pixels = pixels.reshape(-1, pixels.shape[2]) / 255.
    # print(pixels.shape)
    # print(numpy.unique(pixels, axis=0).shape)
    colors, _ = scipy.cluster.vq.kmeans2(pixels, 16)
    colors_srgb = [f'#{r:x}{g:x}{b:x}' for r, g, b in numpy.round(colorspacious.cspace_convert(colors, 'sRGB1', 'sRGB255')).astype(int) if re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', f'#{r:x}{g:x}{b:x}')]
    matplotlib.pyplot.imshow(image)
    matplotlib.pyplot.show()

    return {
        'colors': {
            'background': colors_srgb[0],
            'foreground': colors_srgb[1],
            'foreground-accent': colors_srgb[2],
            'foreground-accent-alt1': colors_srgb[3],
            'foreground-accent-alt2': colors_srgb[4],
            'foreground-accent-complementary': colors_srgb[5],
            'background-accent': colors_srgb[6],
        },
        'fonts': {
            'standard': 'FiraCode Nerd Font',
        }
}



if __name__ == '__main__':
    # generate_debug_preview(theme)
    image_file_path = os.path.expanduser(os.path.join('~', 'Downloads', 'pexels-joshua-135018.jpg'))
    theme = extract_colors_from_image(image_file_path)
    generate_debug_preview(theme)