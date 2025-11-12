from PIL import ImageDraw
from PIL import Image


def circular_crop(img, radius_ratio=0.4):
    w, h = img.size
    r = int(min(w, h) * radius_ratio)
    cx, cy = w // 2, h // 2

    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)

    result = Image.new("RGB", (w, h))
    result.paste(img, mask=mask)
    return result  # Return the masked (circularly cropped) image