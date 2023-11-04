import os
import logging
import collections
import platform
import json

logger = logging.getLogger(__name__)

import PIL.Image
import PIL.ImageFilter
import PIL.ImageDraw
import PIL.ImageFont
import numpy
import scipy.spatial.distance
import sklearn.cluster
import colorspacious
from matplotlib import font_manager
from cpuinfo import get_cpu_info

def extract_colors(pixel_array, n_colors=256):
    model = sklearn.cluster.MiniBatchKMeans(n_clusters=n_colors, n_init="auto", random_state=2106)
    model.fit(pixel_array)
    colors = list()
    for color in model.cluster_centers_:
        colors.append(colorspacious.cspace_convert(color, "sRGB255", "CIELCh"))
    return sorted(colors, key=lambda x: -x[0])


def get_luminance(color):
    color = [x / 255.0 for x in color]
    for i in range(3):
        color[i] = color[i] / 12.92 if color[i] <= 0.03928 else ((color[i] + 0.055) / 1.055) ** 2.4
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


def check_contrast(colors, min_contrast=7):
    valid_colors = collections.defaultdict(list)
    for color_target in colors:
        for color_candidate in colors:
            L_target = get_luminance(colorspacious.cspace_convert(color_target, "CIELCh", "sRGB255"))
            L_candidate = get_luminance(colorspacious.cspace_convert(color_candidate, "CIELCh", "sRGB255"))
            L1 = max(L_target, L_candidate)
            L2 = min(L_target, L_candidate)
            if (L1 + 0.05) / (L2 + 0.05) >= min_contrast:
                valid_colors[tuple(color_target)].append(tuple(color_candidate))
    return dict(sorted(valid_colors.items(), key=lambda x: -len(x[1])))


def select_colors(valid_colors, n_colors=8):
    color_to_n = collections.defaultdict(int)
    n_to_colors = collections.defaultdict(list)
    for color in valid_colors:
        color_to_n[color] = len(valid_colors[color])
        n_to_colors[len(valid_colors[color])].append(color)

    sorted_n = sorted(n_to_colors.keys(), key=lambda x: -x)

    selected_colors = set()
    candidates = set(valid_colors.keys())
    for n in sorted_n:
        if len(selected_colors) > n_colors:
            continue
        for color in n_to_colors[n]:
            if len(selected_colors) > n_colors:
                continue
            if color in selected_colors:
                continue
            if color in candidates:
                selected_colors.add(color)
                candidates = set(valid_colors[color])

    return sorted(selected_colors, key=lambda x: -x[0])


def load_image(input_file_path, n_colors=256):
    image = PIL.Image.open(input_file_path)
    image_rgb = image.convert("RGB")
    image_array = numpy.array(image_rgb)
    pixel_array = image_array.reshape(-1, image_array.shape[2])
    # values, counts = numpy.unique(pixel_array, return_counts=True, axis=0)
    height, width, channels = image_array.shape
    logger.info("image size: %sx%s px", width, height)
    # logger.info("unique colors: %s", len(counts))
    # if len(counts) < n_colors:
    #     raise ValueError("Image has less than 256 colors.")
    return image_rgb, pixel_array


def generate_debug_image(image, colors, selected_colors, base_color, complementary_color, accent_color, variants, color_variations, color_variations_toned, square_colors, blackwhite, blackwhite_toned, theme):
    debug_image = image.resize((1920, 1080))
    debug_image = debug_image.filter(PIL.ImageFilter.GaussianBlur(radius=32))
    canvas = PIL.ImageDraw.Draw(debug_image)

    n_colors = len(colors)
    offset_x = (1920 - (4 * n_colors + (n_colors - 2) + 6)) / 2
    offset_y = 32
    canvas.rectangle(
        [
            (offset_x, offset_y),
            (offset_x + ((4 * n_colors) + (n_colors - 2) + 6), offset_y + (3 + 6)),
        ],
        fill=(17, 17, 17),
    )
    for i, color in enumerate(colors):
        x = offset_x + 3 + i * 5
        y = offset_y + 3
        canvas.rectangle(
            [(x, y), (x + 3, y + 3)],
            fill=tuple(
                map(
                    int,
                    numpy.round(
                        colorspacious.cspace_convert(color, "CIELCh", "sRGB255")
                    ),
                )
            ),
        )

    n_colors = len(selected_colors)
    #offset_x = (1920 - (32 * n_colors + (n_colors - 2) + 6)) / 2
    offset_y = 128
    canvas.rectangle(
        [
            (offset_x, offset_y),
            (offset_x + ((32 * n_colors) + (n_colors - 2) + 6), offset_y + (63 + 6)),
        ],
        fill=(17, 17, 17),
    )
    for i, color in enumerate(selected_colors):
        x = offset_x + 3 + i * 33
        y = offset_y + 3
        # if i == base_color:
        #     outline = (255, 0, 0)
        # elif i == complementary_color:
        #     outline = (0, 0, 255)
        # elif i == accent_color:
        #     outline = (0, 255, 0)
        # else:
        #     outline = None
        outline = None
        canvas.rectangle(
            [(x, y), (x + 31, y + 63)],
            fill=tuple(
                map(
                    int,
                    numpy.round(
                        colorspacious.cspace_convert(color, "CIELCh", "sRGB255")
                    ),
                )
            ),
            outline=outline,
        )

    n_colors = len(variants)
    #offset_x = (1920 - (32 * n_colors + (n_colors - 2) + 6)) / 2
    offset_y = 256
    canvas.rectangle(
        [
            (offset_x, offset_y),
            (offset_x + ((32 * n_colors) + (n_colors - 2) + 6), offset_y + (63 + 6)),
        ],
        fill=(17, 17, 17),
    )
    for i, color in enumerate(variants):
        x = offset_x + 3 + i * 33
        y = offset_y + 3
        canvas.rectangle(
            [(x, y), (x + 31, y + 63)],
            fill=tuple(
                map(
                    int,
                    numpy.round(
                        colorspacious.cspace_convert(color, "CIELCh", "sRGB255")
                    ),
                )
            ),
            outline=None,
        )

    n_colors = len(color_variations)
    #offset_x = (1920 - (32 * n_colors + (n_colors - 2) + 6)) / 2
    offset_y = 256 + 128
    canvas.rectangle(
        [
            (offset_x, offset_y),
            (offset_x + ((32 * n_colors) + (n_colors - 2) + 6), offset_y + (63 + 6)),
        ],
        fill=(17, 17, 17),
    )
    for i, color in enumerate(color_variations):
        x = offset_x + 3 + i * 33
        y = offset_y + 3
        canvas.rectangle(
            [(x, y), (x + 31, y + 63)],
            fill=tuple(
                map(
                    int,
                    numpy.round(
                        colorspacious.cspace_convert(color, "CIELCh", "sRGB255")
                    ),
                )
            ),
            outline=None,
        )
        canvas.rectangle(
            [(x, y + 31), (x + 31, y + 63)],
            fill=tuple(
                map(
                    int,
                    numpy.round(
                        colorspacious.cspace_convert(color_variations_toned[i], "CIELCh", "sRGB255")
                    ),
                )
            ),
            outline=None,
        )

    n_colors = len(square_colors)
    #offset_x = (1920 - (32 * n_colors + (n_colors - 2) + 6)) / 2
    offset_y = 256 + 256
    canvas.rectangle(
        [
            (offset_x, offset_y),
            (offset_x + ((32 * n_colors) + (n_colors - 2) + 6), offset_y + (63 + 6)),
        ],
        fill=(17, 17, 17),
    )
    for i, color in enumerate(square_colors):
        x = offset_x + 3 + i * 33
        y = offset_y + 3
        canvas.rectangle(
            [(x, y), (x + 31, y + 63)],
            fill=tuple(
                map(
                    int,
                    numpy.round(
                        colorspacious.cspace_convert(color, "CIELCh", "sRGB255")
                    ),
                )
            ),
            outline=None,
        )

    n_colors = len(blackwhite)
    #offset_x = (1920 - (32 * n_colors + (n_colors - 2) + 6)) / 2
    offset_y = 256 + 256 + 128
    canvas.rectangle(
        [
            (offset_x, offset_y),
            (offset_x + ((32 * n_colors) + (n_colors - 2) + 6), offset_y + (63 + 6)),
        ],
        fill=(17, 17, 17),
    )
    for i, color in enumerate(blackwhite):
        x = offset_x + 3 + i * 33
        y = offset_y + 3
        canvas.rectangle(
            [(x, y), (x + 31, y + 63)],
            fill=tuple(
                map(
                    int,
                    numpy.round(
                        colorspacious.cspace_convert(color, "CIELCh", "sRGB255")
                    ),
                )
            ),
            outline=None,
        )
        canvas.rectangle(
            [(x, y + 31), (x + 31, y + 63)],
            fill=tuple(
                map(
                    int,
                    numpy.round(
                        colorspacious.cspace_convert(blackwhite_toned[i], "CIELCh", "sRGB255")
                    ),
                )
            ),
            outline=None,
        )

    y = 64 + 32
    x = 1920 // 2
    canvas.rectangle(
        [(x, y + 31), (x + 1920 // 3, y + 1080 // 2.5)],
        fill=theme['colors']['background'],
        outline=None,
    )

    font_file_standard = font_manager.findfont(theme['fonts']['standard'])
    font_standard = PIL.ImageFont.truetype(font_file_standard, 16)
    font_file_console = font_manager.findfont(theme['fonts']['standard'])
    font_console = PIL.ImageFont.truetype(font_file_console, 16)

    canvas.text((x + 256 - 64 + 16, y + 48), 'Hello World!', font=font_standard, fill=theme['colors']['foreground'])

    canvas.text((x + 256 - 64 + 16, y + 48 + 16 + 3), 'OS:', font=font_standard, fill=theme['colors']['highlight'])
    canvas.text((x + 256 - 64 + 16 + 64, y + 48 + 16 + 3), f'{platform.system()} {platform.release()}', font=font_standard, fill=theme['colors']['foreground'])

    canvas.text((x + 256 - 64 + 16, y + 48 + 16 + 16 +6), 'CPU:', font=font_standard, fill=theme['colors']['highlight'])
    canvas.text((x + 256 - 64 + 16 + 64, y + 48 + 16 + 16 +6), f'{get_cpu_info()["brand_raw"]}', font=font_standard, fill=theme['colors']['foreground'])

    canvas.text((x + 256 - 64 + 16, y + 48 + 16 + 16 + 16 + 16 + 12), f'{os.getlogin()}@{platform.node()} ~', font=font_standard, fill=theme['colors']['background-02'])
    canvas.text((x + 256 - 64 + 16, y + 48 + 16 + 16 + 16 + 16 + 16 + 15), f'Sphinx of black quartz, judge my vow.', font=font_standard, fill=theme['colors']['highlight-00'])


    canvas.text((x + 256 - 64 + 16, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21), f'bright', font=font_standard, fill=theme['colors']['bright-red'])
    canvas.text((x + 256 - 64 + 16 + 64, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21), f'red', font=font_standard, fill=theme['colors']['red'])
    canvas.text((x + 256 - 64 + 16, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 19), f'bright', font=font_standard, fill=theme['colors']['bright-green'])
    canvas.text((x + 256 - 64 + 16 + 64, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 19), f'green', font=font_standard, fill=theme['colors']['green'])
    canvas.text((x + 256 - 64 + 16, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 38), f'bright', font=font_standard, fill=theme['colors']['bright-blue'])
    canvas.text((x + 256 - 64 + 16 + 64, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 38), f'blue', font=font_standard, fill=theme['colors']['blue'])

    canvas.text((x + 256 - 64 + 16 + 128, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21), f'bright', font=font_standard, fill=theme['colors']['bright-cyan'])
    canvas.text((x + 256 - 64 + 16 + 64 + 128, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21), f'cyan', font=font_standard, fill=theme['colors']['cyan'])
    canvas.text((x + 256 - 64 + 16 + 128, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 19), f'bright', font=font_standard, fill=theme['colors']['bright-magenta'])
    canvas.text((x + 256 - 64 + 16 + 64 + 128, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 19), f'magenta', font=font_standard, fill=theme['colors']['magenta'])
    canvas.text((x + 256 - 64 + 16 + 128, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 38), f'bright', font=font_standard, fill=theme['colors']['bright-yellow'])
    canvas.text((x + 256 - 64 + 16 + 64 + 128, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 38), f'yellow', font=font_standard, fill=theme['colors']['yellow'])
    
    canvas.text((x + 256 - 64 + 16, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 38 + 64), f'░░░', font=font_standard, fill=theme['colors']['bright-white'])
    canvas.text((x + 256 - 64 + 16 + 64, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 38 + 64), f'▒▒▒', font=font_standard, fill=theme['colors']['white'])
    canvas.text((x + 256 - 64 + 16 + 128, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 38 + 64), f'░░░', font=font_standard, fill=theme['colors']['bright-black'])
    canvas.text((x + 256 - 64 + 16 + 64 + 128, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 38 + 64), f'▒▒▒', font=font_standard, fill=theme['colors']['black'])

    max_size = (256, 256)  # Change this to your desired size

    # Resize the source image to fit within the maximum size while preserving the aspect ratio
    image.thumbnail(max_size)

    # Calculate the coordinates for the middle square portion
    left = (image.width - min(image.width, image.height)) // 2
    top = (image.height - min(image.width, image.height)) // 2
    right = left + min(image.width, image.height)
    bottom = top + min(image.width, image.height)

    # Crop the middle square portion
    cropped_image = image.crop((left, top, right, bottom))

    debug_image.paste(cropped_image, (1920 // 2 + 16, 64 + 64 + 16))


    debug_image.save(os.path.join("data", "debug.png"))


def resize_image_with_crop(input_image_path, output_image_path, size):
    original_image = PIL.Image.open(input_image_path)

    # Calculate the aspect ratio
    original_width, original_height = original_image.size
    target_width, target_height = size
    aspect_ratio = original_width / original_height

    # Calculate the new dimensions
    if (target_width / target_height) > aspect_ratio:
        # The target size is wider than the original image
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
    else:
        # The target size is taller than the original image
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    # Resize the image while preserving the aspect ratio
    resized_image = original_image.resize((new_width, new_height), PIL.Image.ANTIALIAS)

    # Calculate the cropping box
    left = (new_width - target_width) / 2
    top = (new_height - target_height) / 2
    right = (new_width + target_width) / 2
    bottom = (new_height + target_height) / 2

    # Crop the image
    cropped_image = resized_image.crop((left, top, right, bottom))

    # Save the final cropped and resized image
    cropped_image.save(output_image_path)


def determine_base_color(pixel_array, selected_colors):
    selected_colors_sRGB = colorspacious.cspace_convert(selected_colors, "CIELCh", "sRGB255")
    distances = scipy.spatial.distance.cdist(pixel_array, selected_colors_sRGB)
    indices = numpy.argmin(distances, axis=1)
    values, counts = numpy.unique(indices, return_counts=True)
    return values[numpy.argmax(counts)]


def determine_complementary_color(selected_colors, base_color):
    max_contrast = 0
    complementary_color = None
    L_target = get_luminance(colorspacious.cspace_convert(selected_colors[base_color], "CIELCh", "sRGB255"))
    for i, color in enumerate(selected_colors):
        L_candidate = get_luminance(colorspacious.cspace_convert(color, "CIELCh", "sRGB255"))
        L1 = max(L_target, L_candidate)
        L2 = min(L_target, L_candidate)
        if (L1 + 0.05) / (L2 + 0.05) >= max_contrast:
            max_contrast = (L1 + 0.05) / (L2 + 0.05)
            complementary_color = i
    return complementary_color


def determine_accent_color(select_colors, base_color, complementary_color, min_contrast=7):
    candidates = set()
    L_target = get_luminance(colorspacious.cspace_convert(selected_colors[base_color], "CIELCh", "sRGB255"))
    for i, color in enumerate(selected_colors):
        if i == base_color or i == complementary_color:
            continue
        L_candidate = get_luminance(colorspacious.cspace_convert(color, "CIELCh", "sRGB255"))
        L1 = max(L_target, L_candidate)
        L2 = min(L_target, L_candidate)
        if (L1 + 0.05) / (L2 + 0.05) >= min_contrast:
            candidates.add(tuple([i, color]))

    return sorted(candidates, key=lambda x: -x[1][1])[0][0]


def find_base_color_variants(selected_colors, base_color, complementary_color, min_contrast=7, steps=3):
    color_base = list(selected_colors[base_color])
    start_L = color_base[0]
    color_complementary = list(selected_colors[complementary_color])
    L_base = get_luminance(colorspacious.cspace_convert(color_base, "CIELCh", "sRGB255"))
    L_complementary = get_luminance(colorspacious.cspace_convert(color_complementary, "CIELCh", "sRGB255"))
    direction = 0
    if L_base > L_complementary:
        logger.info('light color theme')
        direction = -1
    else:
        logger.info('dark color theme')
        direction = 1

    L1 = max(L_base, L_complementary)
    L2 = min(L_base, L_complementary)
    while (L1 + 0.05) / (L2 + 0.05) >= min_contrast:
        color_base[0] += direction
        L_base = get_luminance(colorspacious.cspace_convert(color_base, "CIELCh", "sRGB255"))
        L1 = max(L_base, L_complementary)
        L2 = min(L_base, L_complementary)
    end_L = color_base[0]

    variants = set()
    variant = color_base.copy()
    L1 = min(start_L, end_L)
    L2 = max(start_L, end_L)
    step_size = (L2 - L1) / (steps + 1)
    for L in numpy.arange(L1, L2, step_size):
        variant[0] = L
        variants.add(tuple(variant))

    return direction, sorted(variants, key=lambda x: direction * x[0])


def calculate_color_variations(accent_color, direction, min_contrast=7):
    L, C, h = accent_color
    variants = list()
    variants_toned = list()

    for i in range(0, 6):
        new_variant = (L, C, (h + i * 60) % 360)
        variants.append(new_variant)
        new_variant_sRGB = colorspacious.cspace_convert(new_variant, "CIELCh", "sRGB255")
        new_variant_toned = [L - direction, C, (h + i * 60) % 360]
        new_variant_toned_sRGB = colorspacious.cspace_convert(new_variant_toned, "CIELCh", "sRGB255")
        L_base = get_luminance(new_variant_sRGB)
        L_complementary = get_luminance(new_variant_toned_sRGB)
        L1 = max(L_base, L_complementary)
        L2 = min(L_base, L_complementary)
        while (L1 + 0.05) / (L2 + 0.05) < min_contrast:
            new_variant_toned[0] -= direction
            new_variant_toned_sRGB = colorspacious.cspace_convert(new_variant_toned, "CIELCh", "sRGB255")
            L_complementary = get_luminance(new_variant_toned_sRGB)
            L1 = max(L_base, L_complementary)
            L2 = min(L_base, L_complementary)
        variants_toned.append(tuple(new_variant_toned))

    return sorted(variants, key=lambda x: x[2]), sorted(variants_toned, key=lambda x: x[2])


def generate_square_colors(accent_color, complementary_color):
    L, C, h = accent_color
    Lc, Cc, hc = complementary_color
    variants = list()
    for i in range(1, 4):
        new_variant = ((L + Lc) / 2, (C + Cc) / 2, ((h + hc) // 2 + i * 90) % 360)
        variants.append(new_variant)

    return variants


def generate_blackwhite(base_color, complementary_color, direction, min_contrast=7):
    blackwhite = list()
    blackwhite_toned = list()
    # black
    if direction == 1: # dark
        L = base_color[0]
        black = (base_color[0], 0, 0)
    else:
        L = complementary_color[0]
        black = (complementary_color[0], 0, 0)
    blackwhite.append(black)

    black_sRGB = colorspacious.cspace_convert(black, "CIELCh", "sRGB255")
    black_toned = [L - direction, 0, 0]
    black_toned_sRGB = colorspacious.cspace_convert(black_toned, "CIELCh", "sRGB255")
    L_base = get_luminance(black_sRGB)
    L_complementary = get_luminance(black_toned_sRGB)
    L1 = max(L_base, L_complementary)
    L2 = min(L_base, L_complementary)
    while (L1 + 0.05) / (L2 + 0.05) < min_contrast:
        black_toned[0] -= direction
        black_toned_sRGB = colorspacious.cspace_convert(black_toned, "CIELCh", "sRGB255")
        L_complementary = get_luminance(black_toned_sRGB)
        L1 = max(L_base, L_complementary)
        L2 = min(L_base, L_complementary)
    blackwhite_toned.append(tuple(black_toned))

    # white
    if direction == 1: # dark
        L = complementary_color[0]
        black = (complementary_color[0], 0, 0)
    else:
        L = base_color[0]
        black = (base_color[0], 0, 0)
    blackwhite.append(black)

    black_sRGB = colorspacious.cspace_convert(black, "CIELCh", "sRGB255")
    black_toned = [L - direction, 0, 0]
    black_toned_sRGB = colorspacious.cspace_convert(black_toned, "CIELCh", "sRGB255")
    L_base = get_luminance(black_sRGB)
    L_complementary = get_luminance(black_toned_sRGB)
    L1 = max(L_base, L_complementary)
    L2 = min(L_base, L_complementary)
    while (L1 + 0.05) / (L2 + 0.05) < min_contrast:
        black_toned[0] -= direction
        black_toned_sRGB = colorspacious.cspace_convert(black_toned, "CIELCh", "sRGB255")
        L_complementary = get_luminance(black_toned_sRGB)
        L1 = max(L_base, L_complementary)
        L2 = min(L_base, L_complementary)
    blackwhite_toned.append(tuple(black_toned))

    return sorted(blackwhite, key=lambda x: direction * x[0]), sorted(blackwhite_toned, key=lambda x: direction * x[0])



def color_to_str(color):
    color_sRGB = colorspacious.cspace_convert(color, "CIELCh", "sRGB255")
    color_sRGB_i = numpy.round(color_sRGB).astype(int)
    return f'#{min(max(0, color_sRGB_i[0]), 255):02x}{min(max(0, color_sRGB_i[1]), 255):02x}{min(max(0, color_sRGB_i[2]), 255):02x}'

if __name__ == "__main__":
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(
        "%(asctime)s\t%(levelname)s\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)
    input_file_path = os.path.join("data", "sun.webp")
    output_file_path = os.path.join("data", "wallpaper.png")
    resize_image_with_crop(input_file_path, output_file_path, (3840, 2160))
    image_rgb, pixel_array = load_image(input_file_path)
    colors = extract_colors(pixel_array)
    valid_colors = check_contrast(colors)
    selected_colors = select_colors(valid_colors)
    base_color = determine_base_color(pixel_array, selected_colors)
    complementary_color = determine_complementary_color(selected_colors, base_color)
    accent_color = determine_accent_color(selected_colors, base_color, complementary_color)
    direction, variants = find_base_color_variants(selected_colors, base_color, complementary_color)
    color_variations, color_variations_toned = calculate_color_variations(selected_colors[accent_color], direction)
    square_colors = generate_square_colors(selected_colors[accent_color], selected_colors[complementary_color])
    blackwhite, blackwhite_toned = generate_blackwhite(selected_colors[base_color], selected_colors[complementary_color], direction)

    print(color_variations)

    theme = {
        'colors': {
            'background': color_to_str(selected_colors[base_color]),
            'foreground': color_to_str(selected_colors[complementary_color]),
            'highlight': color_to_str(selected_colors[accent_color]),
            'background-00': color_to_str(variants[1]),
            'background-01': color_to_str(variants[2]),
            'background-02': color_to_str(variants[3]),
            'highlight-00': color_to_str(square_colors[0]),
            'highlight-01': color_to_str(square_colors[1]),
            'highlight-02': color_to_str(square_colors[2]),

            'bright-red': color_to_str(color_variations[0]),
            'red': color_to_str(color_variations_toned[0]),
            'bright-green': color_to_str(color_variations[2]),
            'green': color_to_str(color_variations_toned[2]),
            'bright-blue': color_to_str(color_variations[4]),
            'blue': color_to_str(color_variations_toned[4]),
            'bright-cyan': color_to_str(color_variations[3]),
            'cyan': color_to_str(color_variations_toned[3]),
            'bright-magenta': color_to_str(color_variations[5]),
            'magenta': color_to_str(color_variations_toned[5]),
            'bright-yellow': color_to_str(color_variations[1]),
            'yellow': color_to_str(color_variations_toned[1]),

            'bright-white': color_to_str(blackwhite[1]),
            'white': color_to_str(blackwhite_toned[1]),
            'bright-black': color_to_str(blackwhite[0]),
            'black': color_to_str(blackwhite_toned[0]),
        },
        'fonts': {
            'standard': 'FiraMono Nerd Font Mono',
            'console': 'FiraMono Nerd Font Mono',
            'variants': 'FiraMonoNFM-Regular +cv16 +ss02',
        },
        'wallpaper': 'stretched',
    }
    json.dump(theme, open(os.path.join('data', 'theme.json'), 'w'), indent=4)
    generate_debug_image(image_rgb, colors, selected_colors, base_color, complementary_color, accent_color, variants, color_variations, color_variations_toned, square_colors, blackwhite, blackwhite_toned, theme)
