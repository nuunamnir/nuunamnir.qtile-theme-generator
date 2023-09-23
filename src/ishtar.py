#!/usr/bin/env python
import os
import io
import argparse
import logging
import json

import PIL.Image
import numpy
from lxml import etree
import colorspacious


def color_to_str(color, colorspace):
    color_sRGB255 = numpy.round(
        colorspacious.cspace_convert(color, colorspace, "sRGB255")
    ).astype(int)
    return f"#{max(0, min(color_sRGB255[0], 255)):02x}{max(0, min(color_sRGB255[1], 255)):02x}{max(0, min(color_sRGB255[2], 255)):02x}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="isthar",
        description="generates a theme for the nuunamnir.dot-files",
    )

    parser.add_argument(
        "-n",
        dest="name",
        type=str,
        help="the name of the theme, if not provided a UUID is generate",
        required=False,
        default=None,
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
        "-o",
        "--output",
        dest="output",
        type=str,
        help="output directory path",
        required=False,
        default="",
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
        "--font-features",
        type=str,
        dest="font_features",
        help="font features for the theme",
        required=False,
        default="FiraMonoNFM-Regular +cv16 +ss02",
    )

    parser.add_argument(
        dest="input",
        type=str,
        help="an input image based on which the theme is generated",
    )

    args = parser.parse_args()

    if args.debug:
        logger = logging.getLogger()
        logger.setLevel(logging.ERROR)

    image_file_path = args.input
    image = PIL.Image.open(image_file_path).convert("RGB")

    image_resized = image.resize((128, 128), resample=PIL.Image.LANCZOS)
    pixels = numpy.array(image_resized.quantize(4, PIL.Image.MEDIANCUT).convert("RGB"))
    pixels = pixels.reshape(-1, pixels.shape[2]) / 255.0

    unique_colors, counts = numpy.unique(pixels, axis=0, return_counts=True)
    idx = numpy.argmax(counts)

    # rule 0: find most frequent color, this is the main theme color
    main_color = unique_colors[idx]
    main_color_sRGB255 = numpy.round(
        colorspacious.cspace_convert(main_color, "sRGB1", "sRGB255")
    ).astype(int)
    main_color_str = f"#{main_color_sRGB255[0]:02x}{main_color_sRGB255[1]:02x}{main_color_sRGB255[2]:02x}"

    # rule 1: identify if main color is "dark" or "light"
    L, C, h = colorspacious.cspace_convert(main_color, "sRGB1", "CIELCh")
    if L > 50:
        theme_mode = "light"
    else:
        theme_mode = "dark"

    # rule 2: identify the main color hue
    main_hues = {
        "red1": 0,
        "yellow": 60,
        "green": 120,
        "cyan": 180,
        "blue": 240,
        "purple": 300,
        "red2": 360,
    }

    min_d = numpy.inf
    min_hue = None
    for hue in main_hues.keys():
        d = abs(h - main_hues[hue])
        if d < min_d:
            min_d = d
            min_hue = hue

    which_red = None
    main_hue_diff = main_hues[min_hue] - h

    # rule 3: generate "black" and "white" versions of the main color
    main_color_black = (10, C, h)
    main_color_bright_black = (15, C, h)
    main_color_white = (85, C, h)
    main_color_bright_white = (90, C, h)

    main_color_mid = (50, C, h)
    main_color_mid_a = (50, C, h - 2 * main_hue_diff)
    main_color_mid_b = (50, C, h + 2 * main_hue_diff)

    main_color_alert = (L, 100, h)
    main_color_alert_a = (L, 100, h - 2 * main_hue_diff)
    main_color_alert_b = (L, 100, h + 2 * main_hue_diff)

    nsmap = {
        "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
        "cc": "http://web.resource.org/cc/",
        "svg": "http://www.w3.org/2000/svg",
        "dc": "http://purl.org/dc/elements/1.1/",
        "xlink": "http://www.w3.org/1999/xlink",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "inkscape": "http://www.inkscape.org/namespaces/inkscape",
    }

    template_file_path = os.path.join("template_preview.svg")
    data = etree.parse(template_file_path)

    data.xpath('//svg:circle[@id="dark_red_circle"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_mid, 'CIELCh')};"
    data.xpath('//svg:circle[@id="dark_green_circle"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_mid_a, 'CIELCh')};"
    data.xpath('//svg:circle[@id="dark_blue_circle"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_mid_b, 'CIELCh')};"

    data.xpath('//svg:circle[@id="light_red_circle"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_mid, 'CIELCh')};"
    data.xpath('//svg:circle[@id="light_green_circle"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_mid_a, 'CIELCh')};"
    data.xpath('//svg:circle[@id="light_blue_circle"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_mid_b, 'CIELCh')};"

    data.xpath('//svg:ellipse[@id="dark_yellow_circle"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_alert, 'CIELCh')};"
    data.xpath('//svg:ellipse[@id="dark_purple_circle"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_alert_a, 'CIELCh')};"
    data.xpath('//svg:ellipse[@id="dark_cyan_circle"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_alert_b, 'CIELCh')}"

    data.xpath('//svg:ellipse[@id="light_yellow_circle"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_alert, 'CIELCh')};"
    data.xpath('//svg:ellipse[@id="light_purple_circle"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_alert_a, 'CIELCh')};"
    data.xpath('//svg:ellipse[@id="light_cyan_circle"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_alert_b, 'CIELCh')}"

    hue_dark_dict = {}
    for hue in main_hues.keys():
        if hue == "red1" and min_hue == "red2" or hue == "red2" and min_hue == "red1":
            continue

        if theme_mode == "dark":
            main_color_hue = (L, C, main_hues[hue] - main_hue_diff)
            main_color_bright_hue = (L + 30, C, main_hues[hue] - main_hue_diff)
        else:
            main_color_hue = (L - 30, C, main_hues[hue] - main_hue_diff)
            main_color_bright_hue = (L, C, main_hues[hue] - main_hue_diff)

        if hue == "red1" or hue == "red2":
            hue = "red"

        try:
            data.xpath(f'//svg:rect[@id="dark_{hue}"]', namespaces=nsmap)[0].attrib[
                "style"
            ] = f"fill:{color_to_str(main_color_hue, 'CIELCh')};"
            data.xpath(
                f'//svg:text[@id="dark_{hue}_text"]/svg:tspan', namespaces=nsmap
            )[0].attrib["style"] = f"fill:{color_to_str(main_color_hue, 'CIELCh')};"

            data.xpath(f'//svg:rect[@id="dark_bright_{hue}"]', namespaces=nsmap)[
                0
            ].attrib["style"] = f"fill:{color_to_str(main_color_bright_hue, 'CIELCh')};"
            data.xpath(
                f'//svg:text[@id="dark_bright_{hue}_text"]/svg:tspan', namespaces=nsmap
            )[0].attrib[
                "style"
            ] = f"fill:{color_to_str(main_color_bright_hue, 'CIELCh')};"

            hue_dark_dict[hue] = color_to_str(main_color_hue, "CIELCh")
            hue_dark_dict[f"bright-{hue}"] = color_to_str(
                main_color_bright_hue, "CIELCh"
            )
        except IndexError:
            logging.error(f"𝍐 | malformed template file")

    hue_light_dict = {}
    for hue in main_hues.keys():
        if hue == "red1" and min_hue == "red2" or hue == "red2" and min_hue == "red1":
            continue

        if theme_mode == "dark":
            main_color_hue = (L, C, main_hues[hue] - main_hue_diff)
            main_color_bright_hue = (L + 30, C, main_hues[hue] - main_hue_diff)
        else:
            main_color_hue = (L - 30, C, main_hues[hue] - main_hue_diff)
            main_color_bright_hue = (L, C, main_hues[hue] - main_hue_diff)

        if hue == "red1" or hue == "red2":
            hue = "red"

        try:
            data.xpath(f'//svg:rect[@id="light_{hue}"]', namespaces=nsmap)[0].attrib[
                "style"
            ] = f"fill:{color_to_str(main_color_bright_hue, 'CIELCh')};"
            data.xpath(
                f'//svg:text[@id="light_{hue}_text"]/svg:tspan', namespaces=nsmap
            )[0].attrib[
                "style"
            ] = f"fill:{color_to_str(main_color_bright_hue, 'CIELCh')};"

            data.xpath(f'//svg:rect[@id="light_bright_{hue}"]', namespaces=nsmap)[
                0
            ].attrib["style"] = f"fill:{color_to_str(main_color_hue, 'CIELCh')};"
            data.xpath(
                f'//svg:text[@id="light_bright_{hue}_text"]/svg:tspan', namespaces=nsmap
            )[0].attrib["style"] = f"fill:{color_to_str(main_color_hue, 'CIELCh')};"

            hue_light_dict[hue] = color_to_str(main_color_bright_hue, "CIELCh")
            hue_light_dict[f"bright-{hue}"] = color_to_str(main_color_hue, "CIELCh")
        except IndexError:
            logging.error(f"𝍐 | malformed template file")

    if theme_mode == "dark":
        data.xpath('//svg:text[@id="title"]/svg:tspan', namespaces=nsmap)[0].attrib[
            "style"
        ] = f"fill:{color_to_str(main_color_white, 'CIELCh')};"
    else:
        data.xpath('//svg:text[@id="title"]/svg:tspan', namespaces=nsmap)[0].attrib[
            "style"
        ] = f"fill:{color_to_str(main_color_black, 'CIELCh')};"
    data.xpath('//svg:text[@id="theme"]/svg:tspan', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_mid, 'CIELCh')};"

    data.xpath('//svg:rect[@id="background"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{main_color_str};"

    data.xpath('//svg:rect[@id="background_dark"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_black, 'CIELCh')};"
    data.xpath('//svg:text[@id="dark"]/svg:tspan', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_bright_white, 'CIELCh')};"

    data.xpath('//svg:rect[@id="background_light"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_bright_white, 'CIELCh')};"
    data.xpath('//svg:text[@id="light"]/svg:tspan', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_black, 'CIELCh')};"

    data.xpath('//svg:rect[@id="dark_bright_white"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_bright_white, 'CIELCh')};"
    data.xpath('//svg:text[@id="dark_bright_white_text"]/svg:tspan', namespaces=nsmap)[
        0
    ].attrib["style"] = f"fill:{color_to_str(main_color_bright_white, 'CIELCh')};"

    data.xpath('//svg:rect[@id="dark_white"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_white, 'CIELCh')};"
    data.xpath('//svg:text[@id="dark_white_text"]/svg:tspan', namespaces=nsmap)[
        0
    ].attrib["style"] = f"fill:{color_to_str(main_color_white, 'CIELCh')};"

    data.xpath('//svg:rect[@id="dark_bright_black"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_bright_black, 'CIELCh')};"
    data.xpath('//svg:text[@id="dark_bright_black_text"]/svg:tspan', namespaces=nsmap)[
        0
    ].attrib["style"] = f"fill:{color_to_str(main_color_bright_black, 'CIELCh')};"

    data.xpath('//svg:rect[@id="dark_black"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_black, 'CIELCh')};"
    data.xpath('//svg:text[@id="dark_black_text"]/svg:tspan', namespaces=nsmap)[
        0
    ].attrib["style"] = f"fill:{color_to_str(main_color_black, 'CIELCh')};"

    data.xpath('//svg:rect[@id="light_bright_white"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_black, 'CIELCh')};"
    data.xpath('//svg:text[@id="light_bright_white_text"]/svg:tspan', namespaces=nsmap)[
        0
    ].attrib["style"] = f"fill:{color_to_str(main_color_black, 'CIELCh')};"

    data.xpath('//svg:rect[@id="light_white"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_bright_black, 'CIELCh')};"
    data.xpath('//svg:text[@id="light_white_text"]/svg:tspan', namespaces=nsmap)[
        0
    ].attrib["style"] = f"fill:{color_to_str(main_color_bright_black, 'CIELCh')};"

    data.xpath('//svg:rect[@id="light_bright_black"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_white, 'CIELCh')};"
    data.xpath('//svg:text[@id="light_bright_black_text"]/svg:tspan', namespaces=nsmap)[
        0
    ].attrib["style"] = f"fill:{color_to_str(main_color_white, 'CIELCh')};"

    data.xpath('//svg:rect[@id="light_black"]', namespaces=nsmap)[0].attrib[
        "style"
    ] = f"fill:{color_to_str(main_color_bright_white, 'CIELCh')};"
    data.xpath('//svg:text[@id="light_black_text"]/svg:tspan', namespaces=nsmap)[
        0
    ].attrib["style"] = f"fill:{color_to_str(main_color_bright_white, 'CIELCh')};"

    if theme_mode == "dark":
        theme = {
            "colors": {
                "base-color": main_color_str,
                "grey1": color_to_str(main_color_mid, "CIELCh"),
                "grey2": color_to_str(main_color_mid_a, "CIELCh"),
                "grey3": color_to_str(main_color_mid_b, "CIELCh"),
                "alert1": color_to_str(main_color_alert, "CIELCh"),
                "alert2": color_to_str(main_color_alert_a, "CIELCh"),
                "alert3": color_to_str(main_color_alert_b, "CIELCh"),
                "black": color_to_str(main_color_black, "CIELCh"),
                "bright-black": color_to_str(main_color_bright_black, "CIELCh"),
                "white": color_to_str(main_color_white, "CIELCh"),
                "bright-white": color_to_str(main_color_bright_white, "CIELCh"),
            },
            "fonts": {
                "standard": args.font_standard,
                "console": args.font_console,
                "features": args.font_features,
            },
            "wallpaper": args.wallpaper_mode,
        }

        for hue in hue_dark_dict:
            theme["colors"][hue] = hue_dark_dict[hue]
    else:
        theme = {
            "colors": {
                "base-color": main_color_str,
                "grey1": color_to_str(main_color_mid, "CIELCh"),
                "grey2": color_to_str(main_color_mid_a, "CIELCh"),
                "grey3": color_to_str(main_color_mid_b, "CIELCh"),
                "alert1": color_to_str(main_color_alert, "CIELCh"),
                "alert2": color_to_str(main_color_alert_a, "CIELCh"),
                "alert3": color_to_str(main_color_alert_b, "CIELCh"),
                "black": color_to_str(main_color_bright_white, "CIELCh"),
                "bright-black": color_to_str(main_color_white, "CIELCh"),
                "white": color_to_str(main_color_bright_black, "CIELCh"),
                "bright-white": color_to_str(main_color_bright_black, "CIELCh"),
            },
            "fonts": {
                "standard": args.font_standard,
                "console": args.font_console,
                "features": args.font_features,
            },
            "wallpaper": args.wallpaper_mode,
        }

        for hue in hue_light_dict:
            theme["colors"][hue] = hue_light_dict[hue]

    wallpaper_width, wallpaper_height = args.width, args.height
    wallpaper = PIL.Image.new(
        size=(wallpaper_width, wallpaper_height), mode="RGB", color=main_color_str
    )
    wallpaper_image = image.copy()
    wallpaper_image.thumbnail(
        (wallpaper_width, wallpaper_height), PIL.Image.Resampling.LANCZOS
    )
    wallpaper_image_width, wallpaper_image_height = wallpaper_image.size
    offset = (
        (wallpaper_width - wallpaper_image_width) // 2,
        (wallpaper_height - wallpaper_image_height) // 2,
    )
    wallpaper.paste(wallpaper_image, offset)
    if args.wallpaper_mode not in ["stretched", "centered", "tiled"]:
        logging.error(
            f"𝍐 | the provided wallpaper mode is not supported, only streteched, centered and tiled are supported"
        )
        raise ValueError

    try:
        theme_dir_path = os.path.join(args.output, f"{theme_mode}_{args.name}")
        os.makedirs(theme_dir_path)
        with io.open(
            os.path.join(theme_dir_path, "preview.svg"), "w", encoding="utf-8"
        ) as output_handle:
            output_handle.write(etree.tostring(data, pretty_print=True).decode("utf-8"))

        with io.open(
            os.path.join(theme_dir_path, "theme.json"), "w", encoding="utf-8"
        ) as output_handle:
            json.dump(theme, output_handle)

        wallpaper.save(os.path.join(theme_dir_path, "wallpaper.png"))
    except FileExistsError:
        logging.error(f"𝍐 | a theme of this name {args.name} already exists")
