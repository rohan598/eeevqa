

def render_text_on_bounding_box(
    text: str,
    bounding_box: Iterable[Iterable[int]],
    image: Image.Image,
    font_path: str):
    
    """Render text on top of a specific bounding box."""
    draw = ImageDraw.Draw(image)
    (x0, y0), (x1, y1) = bounding_box
    
    draw.rectangle(xy=[(x0, y0), (x1, y1)], fill=(255, 255, 255, 255))
    
    fontsize = 1
    def _can_increment_font(ratio=0.95):
        next_font = ImageFont.truetype(
            font_path, encoding="UTF-8", size=fontsize + 1)
        width, height = next_font.getsize(text)
        return width < ratio * (x1 - x0) and height < ratio * (y1 - y0)

    while _can_increment_font():
        fontsize += 1
    font = ImageFont.truetype(font_path, encoding="UTF-8", size=fontsize)

    draw.text(
        xy=((x0 + x1)/2, (y0 + y1)/2),
        text=text,
        font=font,
        fill="black",
        anchor="mm"
    )
    
def render_text(text: str,
                text_size: int = 36,
                text_color: str = "black",
                background_color: str = "white",
                left_padding: int = 5,
                right_padding: int = 5,
                top_padding: int = 5,
                bottom_padding: int = 5,
                font_path: str = "") -> Image.Image:

    """Render text."""
    # Add new lines so that each line is no more than 80 characters.
    wrapper = textwrap.TextWrapper(width=80)
    lines = wrapper.wrap(text=text)
    wrapped_text = "\n".join(lines)

    font = ImageFont.truetype(font_path, encoding="UTF-8", size=text_size)

    # Use a temporary canvas to determine the width and height in pixels when
    # rendering the text.
    temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1), background_color))
    _, _, text_width, text_height = temp_draw.textbbox((0, 0), wrapped_text, font)

    # Create the actual image with a bit of padding around the text.
    image_width = text_width + left_padding + right_padding
    image_height = text_height + top_padding + bottom_padding
    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)
    draw.text(
      xy=(left_padding, top_padding),
      text=wrapped_text,
      fill=text_color,
      font=font)
    
    return image


def render_header(image: Image.Image, header: str, font_path: str) -> Image.Image:
    """Renders a header on a PIL image and returns a new PIL image."""
    header_image = render_text(header, font_path=font_path)
    new_width = max(header_image.width, image.width)

    new_height = int(image.height *  (new_width / image.width))
    new_header_height = int(
        header_image.height * (new_width / header_image.width))

    new_image = Image.new(
        "RGB",
        (new_width, new_height + new_header_height),
        "white")
    new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))
    new_image.paste(image.resize((new_width, new_height)), (0, new_header_height))

    return new_image