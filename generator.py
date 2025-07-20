from PIL import Image, ImageDraw, ImageFont
import os
import textwrap


def text_to_image(
    text: str,
    font_path: str = "",
    font_size: int = 40,
    text_color: str = "black",
    bg_color: str = "white",
    padding: int = 20,
    output_path: str = "output.png",
    max_width: int = None
):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Cannot find font file: {font_path}")
        print("Trying default font...")
        font = ImageFont.load_default()

    temp_img = Image.new("RGB", (1, 1), bg_color)
    temp_draw = ImageDraw.Draw(temp_img)

    if max_width is None:
        text_bbox = temp_draw.textbbox((0, 0), text, font=font)

        width = text_bbox[2] - text_bbox[0] + 2 * padding
        height = text_bbox[3] - text_bbox[1] + 2 * padding

        image = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(image)

        text_x = padding - text_bbox[0]
        text_y = padding - text_bbox[1]

        draw.text((text_x, text_y), text, font=font, fill=text_color)
    else:
        avg_char_width = temp_draw.textlength("x" * 10, font=font) / 10
        chars_per_line = int((max_width - 2 * padding) / avg_char_width)

        wrapped_text = textwrap.fill(text, width=chars_per_line)

        lines = wrapped_text.split("\n")
        line_heights = []
        line_widths = []

        for line in lines:
            bbox = temp_draw.textbbox((0, 0), line, font=font)
            line_heights.append(bbox[3] - bbox[1])
            line_widths.append(bbox[2] - bbox[0])

        max_line_width = max(line_widths)
        total_height = sum(line_heights)
        line_spacing = font_size * 0.5

        width = max_line_width + 2 * padding
        height = total_height + (len(lines) - 1) * line_spacing + 2 * padding

        image = Image.new("RGB", (int(width), int(height)), bg_color)
        draw = ImageDraw.Draw(image)

        current_y = padding
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_x = padding - bbox[0]
            draw.text((text_x, current_y), line, font=font, fill=text_color)
            current_y += line_heights[i] + line_spacing

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        image.save(output_path)
    except Exception as e:
        print(f"Error saving image: {e}")
