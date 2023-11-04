import logging

logger = logging.getLogger(__name__)

import os
import collections
import uuid
import json
import argparse

import platform
from cpuinfo import get_cpu_info

import numpy
import scipy.spatial
import PIL.Image
import PIL.ImageFilter
import PIL.ImageDraw
import PIL.ImageFont
import sklearn.cluster
import colorspacious
from matplotlib import font_manager


def load(image_file_path):
    image = PIL.Image.open(image_file_path).convert("RGB")
    return image


def extract_colors(image, n_colors=256, min_n_colors=8):
    image_array = numpy.array(image)
    pixels = image_array.reshape(-1, image_array.shape[2])
    model = sklearn.cluster.MiniBatchKMeans(
        n_clusters=n_colors, n_init="auto", random_state=2106
    )
    model.fit(pixels)

    colors = list()
    for cluster_center in model.cluster_centers_:
        colors.append(colorspacious.cspace_convert(cluster_center, "sRGB255", "CIELCh"))

    unique_colors = set()
    for color in colors:
        unique_colors.add(tuple(color))

    # TODO: handle images with few unique colors
    if len(unique_colors) < min_n_colors:
        logger.error(
            f"found only {len(unique_colors)} unique colors; at least {n_colors} are required"
        )
        raise ValueError(
            f"found only {len(unique_colors)} unique colors; at least {n_colors} are required"
        )

    return sorted(colors, key=lambda x: -x[0])


def get_luminance(color):
    color = [x / 255.0 for x in color]
    for i in range(3):
        color[i] = (
            color[i] / 12.92
            if color[i] <= 0.03928
            else ((color[i] + 0.055) / 1.055) ** 2.4
        )
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


def check_contrast(colors, min_contrast=7, n_colors=8):
    valid_colors = collections.defaultdict(list)
    contrasts = set()
    contrasts.add(min_contrast)
    i = 0
    while len(valid_colors) < n_colors:
        valid_colors = collections.defaultdict(list)
        loop_min_contrast = sorted(contrasts, reverse=True)[i]
        if loop_min_contrast < min_contrast:
            logger.warning(
                f"lowering minimum contrast to {loop_min_contrast}; this might lead to accessibility issues"
            )
        for color_target in colors:
            for color_candidate in colors:
                L_target = get_luminance(
                    colorspacious.cspace_convert(color_target, "CIELCh", "sRGB255")
                )
                L_candidate = get_luminance(
                    colorspacious.cspace_convert(color_candidate, "CIELCh", "sRGB255")
                )
                L1 = max(L_target, L_candidate)
                L2 = min(L_target, L_candidate)
                contrast = (L1 + 0.05) / (L2 + 0.05)
                contrasts.add(contrast)
                if contrast >= loop_min_contrast:
                    valid_colors[tuple(color_target)].append(tuple(color_candidate))
        i += 1
        logger.debug(
            f"found {len(valid_colors)} colors with a minimum contrast of {loop_min_contrast}"
        )
    return (
        dict(sorted(valid_colors.items(), key=lambda x: -len(x[1]))),
        loop_min_contrast,
    )


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

    if len(selected_colors) < n_colors:
        logger.warning(
            f"could only select {len(selected_colors)} colors; at least {n_colors} are required: augmenting selected colors; this might lead to accessibility issues"
        )
        while len(selected_colors) < n_colors:
            for color in valid_colors:
                if color not in selected_colors:
                    selected_colors.add(color)
                    break

    return sorted(selected_colors, key=lambda x: -x[0])


def determine_base_color(image, selected_colors):
    image_array = numpy.array(image)
    pixels = image_array.reshape(-1, image_array.shape[2])

    selected_colors_sRGB = colorspacious.cspace_convert(
        selected_colors, "CIELCh", "sRGB255"
    )
    distances = scipy.spatial.distance.cdist(pixels, selected_colors_sRGB)
    indices = numpy.argmin(distances, axis=1)
    values, counts = numpy.unique(indices, return_counts=True)
    base_color = values[numpy.argmax(counts)]
    logger.debug(f"determined base color {selected_colors[base_color]}")
    return base_color


def determine_complementary_color(base_color_idx, selected_colors):
    max_contrast = 0
    complementary_color = None
    L_target = get_luminance(
        colorspacious.cspace_convert(
            selected_colors[base_color_idx], "CIELCh", "sRGB255"
        )
    )
    for i, color in enumerate(selected_colors):
        L_candidate = get_luminance(
            colorspacious.cspace_convert(color, "CIELCh", "sRGB255")
        )
        L1 = max(L_target, L_candidate)
        L2 = min(L_target, L_candidate)
        if (L1 + 0.05) / (L2 + 0.05) >= max_contrast:
            max_contrast = (L1 + 0.05) / (L2 + 0.05)
            complementary_color = i
    logger.debug(
        f"determined complementary color {selected_colors[complementary_color]} with contrast {max_contrast}"
    )
    return complementary_color


def determine_accent_color(
    base_color_idx, complementary_color_idx, select_colors, min_contrast=7
):
    candidates = set()
    L_target = get_luminance(
        colorspacious.cspace_convert(
            selected_colors[base_color_idx], "CIELCh", "sRGB255"
        )
    )
    for i, color in enumerate(selected_colors):
        if i == base_color_idx or i == complementary_color_idx:
            continue
        L_candidate = get_luminance(
            colorspacious.cspace_convert(color, "CIELCh", "sRGB255")
        )
        L1 = max(L_target, L_candidate)
        L2 = min(L_target, L_candidate)
        contrast = (L1 + 0.05) / (L2 + 0.05)
        candidates.add(tuple([i, contrast, color]))

    selected_candidate = sorted(candidates, key=lambda x: (-x[1], -x[2][1]))[0]
    accent_color = selected_candidate[0]
    if selected_candidate[1] < min_contrast:
        logger.warning(
            f"could not find an accent color with a contrast of at least {min_contrast}, actual contrast is {selected_candidate[1]}; using the color with the highest contrast {selected_candidate[2]}"
        )
    logger.debug(f"determined accent color {selected_colors[accent_color]}")

    return accent_color


def determine_base_color_variants(
    base_color_idx, complementary_color_idx, selected_colors, min_contrast=7, steps=3
):
    color_base = list(selected_colors[base_color_idx])
    start_L = color_base[0]
    color_complementary = list(selected_colors[complementary_color_idx])
    L_base = get_luminance(
        colorspacious.cspace_convert(color_base, "CIELCh", "sRGB255")
    )
    L_complementary = get_luminance(
        colorspacious.cspace_convert(color_complementary, "CIELCh", "sRGB255")
    )
    direction = 0
    if L_base > L_complementary:
        logger.info("light color theme")
        direction = -1
    else:
        logger.info("dark color theme")
        direction = 1

    L1 = max(L_base, L_complementary)
    L2 = min(L_base, L_complementary)
    while (L1 + 0.05) / (L2 + 0.05) >= min_contrast:
        color_base[0] += direction
        L_base = get_luminance(
            colorspacious.cspace_convert(color_base, "CIELCh", "sRGB255")
        )
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


def determine_accent_color_variants(
    accent_color_idx, selected_colors, direction, min_contrast=7
):
    L, C, h = selected_colors[accent_color_idx]
    variants = list()
    variants_toned = list()

    for i in range(0, 6):
        new_variant = (L, C, (h + i * 60) % 360)
        variants.append(new_variant)
        new_variant_sRGB = colorspacious.cspace_convert(
            new_variant, "CIELCh", "sRGB255"
        )
        new_variant_toned = [L - direction, C, (h + i * 60) % 360]
        new_variant_toned_sRGB = colorspacious.cspace_convert(
            new_variant_toned, "CIELCh", "sRGB255"
        )
        L_base = get_luminance(new_variant_sRGB)
        L_complementary = get_luminance(new_variant_toned_sRGB)
        L1 = max(L_base, L_complementary)
        L2 = min(L_base, L_complementary)
        while (L1 + 0.05) / (L2 + 0.05) < min_contrast:
            new_variant_toned[0] -= direction
            new_variant_toned_sRGB = colorspacious.cspace_convert(
                new_variant_toned, "CIELCh", "sRGB255"
            )
            L_complementary = get_luminance(new_variant_toned_sRGB)
            L1 = max(L_base, L_complementary)
            L2 = min(L_base, L_complementary)
        variants_toned.append(tuple(new_variant_toned))

    return (
        sorted(variants, key=lambda x: x[2]),
        sorted(variants_toned, key=lambda x: x[2]),
    )


def determine_square_color_variants(
    accent_color_idx, complementary_color_idx, selected_colors
):
    L, C, h = selected_colors[accent_color_idx]
    Lc, Cc, hc = selected_colors[complementary_color_idx]
    variants = list()
    for i in range(1, 4):
        new_variant = ((L + Lc) / 2, (C + Cc) / 2, ((h + hc) // 2 + i * 90) % 360)
        variants.append(new_variant)

    return variants


def determine_grey_color_variants(
    base_color_idx, complementary_color_idx, selected_colors, direction, min_contrast=7
):
    base_color = selected_colors[base_color_idx]
    complementary_color = selected_colors[complementary_color_idx]

    blackwhite = list()
    blackwhite_toned = list()
    # black
    if direction == 1:  # dark
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
        black_toned_sRGB = colorspacious.cspace_convert(
            black_toned, "CIELCh", "sRGB255"
        )
        L_complementary = get_luminance(black_toned_sRGB)
        L1 = max(L_base, L_complementary)
        L2 = min(L_base, L_complementary)
    blackwhite_toned.append(tuple(black_toned))

    # white
    if direction == 1:  # dark
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
        black_toned_sRGB = colorspacious.cspace_convert(
            black_toned, "CIELCh", "sRGB255"
        )
        L_complementary = get_luminance(black_toned_sRGB)
        L1 = max(L_base, L_complementary)
        L2 = min(L_base, L_complementary)
    blackwhite_toned.append(tuple(black_toned))

    return (
        sorted(blackwhite, key=lambda x: direction * x[0]),
        sorted(blackwhite_toned, key=lambda x: direction * x[0]),
    )


def resize_image(image, width, height):
    image_width, image_height = image.size
    aspect_ratio = image_width / image_height

    if width / height > aspect_ratio:
        new_width = int(height * aspect_ratio)
        new_height = height
    else:
        new_width = width
        new_height = int(width / aspect_ratio)

    resized_image = image.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)

    left = (new_width - width) / 2
    top = (new_height - height) / 2
    right = (new_width + width) / 2
    bottom = (new_height + height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))

    return cropped_image


def generate_preview(
    image,
    theme,
    colors,
    selected_colors,
    base_color_variants,
    accent_color_variants,
    square_color_variants,
    grey_color_variants,
    width=1920,
    height=1080,
):
    def _draw_swatch(canvas, colors, offset_x, offset_y, swatch_width, swatch_height):
        n_colors = len(colors)
        canvas.rectangle(
            [
                (offset_x, offset_y),
                (
                    offset_x + ((swatch_width * n_colors) + (n_colors - 2) + 6),
                    offset_y + (swatch_height + 6),
                ),
            ],
            fill=(17, 17, 17),
        )
        for i, color in enumerate(colors):
            x = offset_x + 3 + i * (swatch_width + 1)
            y = offset_y + 3
            canvas.rectangle(
                [(x, y), (x + (swatch_width - 1), y + swatch_height)],
                fill=tuple(
                    map(
                        int,
                        numpy.round(
                            colorspacious.cspace_convert(color, "CIELCh", "sRGB255")
                        ),
                    )
                ),
            )

    def _draw_swatch_bipartite(
        canvas, bipartite_colors, offset_x, offset_y, swatch_width, swatch_height
    ):
        colors, colors_tones = bipartite_colors
        n_colors = len(colors)
        canvas.rectangle(
            [
                (offset_x, offset_y),
                (
                    offset_x + ((swatch_width * n_colors) + (n_colors - 2) + 6),
                    offset_y + (swatch_height + 6),
                ),
            ],
            fill=(17, 17, 17),
        )
        for i, color in enumerate(colors):
            x = offset_x + 3 + i * (swatch_width + 1)
            y = offset_y + 3
            canvas.rectangle(
                [(x, y), (x + (swatch_width - 1), y + swatch_height)],
                fill=tuple(
                    map(
                        int,
                        numpy.round(
                            colorspacious.cspace_convert(color, "CIELCh", "sRGB255")
                        ),
                    )
                ),
            )
            canvas.rectangle(
                [
                    (x, y + swatch_height // 2),
                    (x + (swatch_width - 1), y + swatch_height),
                ],
                fill=tuple(
                    map(
                        int,
                        numpy.round(
                            colorspacious.cspace_convert(
                                colors_tones[i], "CIELCh", "sRGB255"
                            )
                        ),
                    )
                ),
            )

    preview_image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
    preview_image = preview_image.filter(PIL.ImageFilter.GaussianBlur(32))

    # TODO: calculate offsets based on preview image dimensions

    canvas = PIL.ImageDraw.Draw(preview_image)
    offset_x = (width - (4 * len(colors) + (len(colors) - 2) + 6)) / 2
    _draw_swatch(canvas, colors, offset_x, 32, 4, 3)
    _draw_swatch(canvas, selected_colors, offset_x, 128, 32, 63)
    _draw_swatch(canvas, base_color_variants, offset_x, 256, 32, 63)
    _draw_swatch_bipartite(canvas, accent_color_variants, offset_x, 256 + 128, 32, 63)
    _draw_swatch(canvas, square_color_variants, offset_x, 512, 32, 63)
    _draw_swatch_bipartite(canvas, grey_color_variants, offset_x, 512 + 128, 32, 63)

    y = 64 + 32
    x = width // 2
    canvas.rectangle(
        [(x, y + 31), (x + width // 3, y + height // 2.5)],
        fill=theme["colors"]["background"],
        outline=None,
    )

    font_file_standard = font_manager.findfont(theme["fonts"]["standard"])
    font_standard = PIL.ImageFont.truetype(font_file_standard, 16)
    font_file_console = font_manager.findfont(theme["fonts"]["standard"])
    font_console = PIL.ImageFont.truetype(font_file_console, 16)

    canvas.text(
        (x + 256 - 64 + 16, y + 48),
        "Hello World!",
        font=font_standard,
        fill=theme["colors"]["foreground"],
    )

    canvas.text(
        (x + 256 - 64 + 16, y + 48 + 16 + 3),
        "OS:",
        font=font_standard,
        fill=theme["colors"]["highlight"],
    )
    canvas.text(
        (x + 256 - 64 + 16 + 64, y + 48 + 16 + 3),
        f"{platform.system()} {platform.release()}",
        font=font_standard,
        fill=theme["colors"]["foreground"],
    )

    canvas.text(
        (x + 256 - 64 + 16, y + 48 + 16 + 16 + 6),
        "CPU:",
        font=font_standard,
        fill=theme["colors"]["highlight"],
    )
    canvas.text(
        (x + 256 - 64 + 16 + 64, y + 48 + 16 + 16 + 6),
        f'{get_cpu_info()["brand_raw"]}',
        font=font_standard,
        fill=theme["colors"]["foreground"],
    )

    canvas.text(
        (x + 256 - 64 + 16, y + 48 + 16 + 16 + 16 + 16 + 12),
        f"{os.getlogin()}@{platform.node()} ~",
        font=font_standard,
        fill=theme["colors"]["background-02"],
    )
    canvas.text(
        (x + 256 - 64 + 16, y + 48 + 16 + 16 + 16 + 16 + 16 + 15),
        f"Sphinx of black quartz, judge my vow.",
        font=font_standard,
        fill=theme["colors"]["highlight-00"],
    )

    canvas.text(
        (x + 256 - 64 + 16, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21),
        f"bright",
        font=font_standard,
        fill=theme["colors"]["bright-red"],
    )
    canvas.text(
        (x + 256 - 64 + 16 + 64, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21),
        f"red",
        font=font_standard,
        fill=theme["colors"]["red"],
    )
    canvas.text(
        (x + 256 - 64 + 16, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 19),
        f"bright",
        font=font_standard,
        fill=theme["colors"]["bright-green"],
    )
    canvas.text(
        (x + 256 - 64 + 16 + 64, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 19),
        f"green",
        font=font_standard,
        fill=theme["colors"]["green"],
    )
    canvas.text(
        (x + 256 - 64 + 16, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 38),
        f"bright",
        font=font_standard,
        fill=theme["colors"]["bright-blue"],
    )
    canvas.text(
        (x + 256 - 64 + 16 + 64, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 38),
        f"blue",
        font=font_standard,
        fill=theme["colors"]["blue"],
    )

    canvas.text(
        (x + 256 - 64 + 16 + 128, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21),
        f"bright",
        font=font_standard,
        fill=theme["colors"]["bright-cyan"],
    )
    canvas.text(
        (x + 256 - 64 + 16 + 64 + 128, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21),
        f"cyan",
        font=font_standard,
        fill=theme["colors"]["cyan"],
    )
    canvas.text(
        (x + 256 - 64 + 16 + 128, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 19),
        f"bright",
        font=font_standard,
        fill=theme["colors"]["bright-magenta"],
    )
    canvas.text(
        (x + 256 - 64 + 16 + 64 + 128, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 19),
        f"magenta",
        font=font_standard,
        fill=theme["colors"]["magenta"],
    )
    canvas.text(
        (x + 256 - 64 + 16 + 128, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 38),
        f"bright",
        font=font_standard,
        fill=theme["colors"]["bright-yellow"],
    )
    canvas.text(
        (x + 256 - 64 + 16 + 64 + 128, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 38),
        f"yellow",
        font=font_standard,
        fill=theme["colors"]["yellow"],
    )

    canvas.text(
        (x + 256 - 64 + 16, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 38 + 64),
        f"░░░",
        font=font_standard,
        fill=theme["colors"]["bright-white"],
    )
    canvas.text(
        (x + 256 - 64 + 16 + 64, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 38 + 64),
        f"▒▒▒",
        font=font_standard,
        fill=theme["colors"]["white"],
    )
    canvas.text(
        (x + 256 - 64 + 16 + 128, y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 38 + 64),
        f"░░░",
        font=font_standard,
        fill=theme["colors"]["bright-black"],
    )
    canvas.text(
        (
            x + 256 - 64 + 16 + 64 + 128,
            y + 48 + 16 + 16 + 16 + 16 + 16 + 32 + 21 + 38 + 64,
        ),
        f"▒▒▒",
        font=font_standard,
        fill=theme["colors"]["black"],
    )

    max_size = (height // 4, height // 4)
    image.thumbnail(max_size)

    left = (image.width - min(image.width, image.height)) // 2
    top = (image.height - min(image.width, image.height)) // 2
    right = left + min(image.width, image.height)
    bottom = top + min(image.width, image.height)

    cropped_image = image.crop((left, top, right, bottom))

    preview_image.paste(cropped_image, (width // 2 + 16, 64 + 64 + 16))

    return preview_image


def save(
    wallpaper,
    preview,
    theme,
    output_directory_path,
    mode,
    theme_name=None,
    overwrite=False,
):
    if theme_name is None:
        theme_name = uuid.uuid4().hex
        logger.warning(f"no theme name provided, using {theme_name} instead")
    theme_directory_path = os.path.expanduser(
        os.path.join(output_directory_path, theme_name)
    )
    try:
        os.makedirs(theme_directory_path)
    except FileExistsError:
        logger.warning(f"a theme of this name already exists")

    try:
        os.makedirs(os.path.join(theme_directory_path, f"{mode}_{theme_name}"))
    except FileExistsError:
        if not overwrite:
            logger.error(
                f"a theme of this name and mode already exists; to overwrite use the --overwrite flag"
            )
            return
        logger.warning(f"a theme of this name and mode already exists; overwriting")

    wallpaper_path = os.path.join(
        theme_directory_path, f"{mode}_{theme_name}", "wallpaper.png"
    )
    wallpaper.save(wallpaper_path, "PNG")

    preview_path = os.path.join(
        theme_directory_path, f"{mode}_{theme_name}", "preview.png"
    )
    preview.save(preview_path, "PNG")

    json.dump(
        theme,
        open(
            os.path.join(theme_directory_path, f"{mode}_{theme_name}", "theme.json"),
            "w",
        ),
        indent=4,
    )


def color_to_str(color):
    color_sRGB = colorspacious.cspace_convert(color, "CIELCh", "sRGB255")
    color_sRGB_i = numpy.round(color_sRGB).astype(int)
    return f"#{min(max(0, color_sRGB_i[0]), 255):02x}{min(max(0, color_sRGB_i[1]), 255):02x}{min(max(0, color_sRGB_i[2]), 255):02x}"


def compile_theme(
    wallpaper_mode,
    fonts,
    selected_colors,
    base_color_idx,
    accent_color_idx,
    complementary_color_idx,
    base_color_variants,
    accent_color_variants,
    square_color_variants,
    grey_color_variants,
):
    theme = {
        "colors": {
            "background": color_to_str(selected_colors[base_color_idx]),
            "foreground": color_to_str(selected_colors[complementary_color_idx]),
            "highlight": color_to_str(selected_colors[accent_color_idx]),
            "background-00": color_to_str(base_color_variants[1]),
            "background-01": color_to_str(base_color_variants[2]),
            "background-02": color_to_str(base_color_variants[3]),
            "highlight-00": color_to_str(square_color_variants[0]),
            "highlight-01": color_to_str(square_color_variants[1]),
            "highlight-02": color_to_str(square_color_variants[2]),
            "bright-red": color_to_str(accent_color_variants[0][0]),
            "red": color_to_str(accent_color_variants[1][0]),
            "bright-green": color_to_str(accent_color_variants[0][2]),
            "green": color_to_str(accent_color_variants[1][2]),
            "bright-blue": color_to_str(accent_color_variants[0][4]),
            "blue": color_to_str(accent_color_variants[1][4]),
            "bright-cyan": color_to_str(accent_color_variants[0][3]),
            "cyan": color_to_str(accent_color_variants[1][3]),
            "bright-magenta": color_to_str(accent_color_variants[0][5]),
            "magenta": color_to_str(accent_color_variants[1][5]),
            "bright-yellow": color_to_str(accent_color_variants[0][1]),
            "yellow": color_to_str(accent_color_variants[1][1]),
            "bright-white": color_to_str(grey_color_variants[0][1]),
            "white": color_to_str(grey_color_variants[1][1]),
            "bright-black": color_to_str(grey_color_variants[0][0]),
            "black": color_to_str(grey_color_variants[1][0]),
        },
        "fonts": fonts,
        "wallpaper": wallpaper_mode,
    }

    return theme


if __name__ == "__main__":
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(
        "%(asctime)s\t%(levelname)s\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    logger.setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(
        prog="inanna",
        description="generates a theme for the nuunamnir environment",
    )

    parser.add_argument(
        "-d",
        "--debug",
        dest="debug",
        help="toggles between debug and production mode",
        required=False,
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        help="output directory path",
        required=False,
        default=".",
    )
    parser.add_argument(
        "-n",
        "--name",
        dest="name",
        type=str,
        help="the name of the theme, if not provided a UUID is generate",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        help="overwrite existing theme data",
        required=False,
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-x",
        "--width",
        dest="width",
        type=int,
        help="target screen resolution width in pixel",
        required=False,
        default=3840,
    )
    parser.add_argument(
        "-y",
        "--height",
        dest="height",
        type=int,
        help="target screen resolution height in pixel",
        required=False,
        default=2160,
    )
    parser.add_argument(
        "-w",
        "--wallpaper",
        dest="wallpaper_mode",
        type=str,
        help="mode in which the image is used as wallpaper in this theme",
        choices=["stretched", "centered", "tiled"],
        required=False,
        default="stretched",
    )

    parser.add_argument(
        "-fs",
        "--font-standard",
        type=str,
        dest="font_standard",
        help="standard font for the theme",
        required=False,
        default="FiraMono Nerd Font Mono",
    )
    parser.add_argument(
        "-fc",
        "--font-console",
        type=str,
        dest="font_console",
        help="console (monospace) font for the theme",
        required=False,
        default="FiraMono Nerd Font Mono",
    )
    parser.add_argument(
        "-ff",
        "--font-variants",
        type=str,
        dest="font_variants",
        help="font variants for the theme",
        required=False,
        default="FiraMonoNFM-Regular +cv16 +ss02",
    )

    parser.add_argument(
        dest="input",
        type=str,
        help="path to an image from which the theme is generated",
    )

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"debug mode enabled")

    image = load(args.input)
    wallpaper = resize_image(image, args.width, args.height)

    colors = extract_colors(wallpaper)
    valid_colors, adjusted_min_contrast = check_contrast(colors)
    selected_colors = select_colors(valid_colors)

    base_color_idx = determine_base_color(wallpaper, selected_colors)
    complementary_color_idx = determine_complementary_color(
        base_color_idx, selected_colors
    )
    accent_color_idx = determine_accent_color(
        base_color_idx, complementary_color_idx, selected_colors
    )

    direction, base_color_variants = determine_base_color_variants(
        base_color_idx,
        complementary_color_idx,
        selected_colors,
        min_contrast=adjusted_min_contrast,
        steps=3,
    )
    accent_color_variants = determine_accent_color_variants(
        accent_color_idx, selected_colors, direction, min_contrast=adjusted_min_contrast
    )

    square_color_variants = determine_square_color_variants(
        accent_color_idx, complementary_color_idx, selected_colors
    )

    grey_color_variants = determine_grey_color_variants(
        base_color_idx,
        complementary_color_idx,
        selected_colors,
        direction,
        min_contrast=7,
    )

    fonts = {
        "standard": args.font_standard,
        "console": args.font_console,
        "variants": args.font_variants,
    }

    theme = compile_theme(
        args.wallpaper_mode,
        fonts,
        selected_colors,
        base_color_idx,
        accent_color_idx,
        complementary_color_idx,
        base_color_variants,
        accent_color_variants,
        square_color_variants,
        grey_color_variants,
    )

    preview = generate_preview(
        wallpaper,
        theme,
        colors,
        selected_colors,
        base_color_variants,
        accent_color_variants,
        square_color_variants,
        grey_color_variants,
    )

    if direction == -1:
        mode = "light"
    else:
        mode = "dark"
    save(wallpaper, preview, theme, args.output, mode, args.name, args.overwrite)
