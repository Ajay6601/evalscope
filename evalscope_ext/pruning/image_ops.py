"""Image degradations for the encoder probe.

The idea: re-ask the same questions with the image gently damaged in ways that
hurt an image encoder first (lower resolution, compression, blur, fewer colours).
A solid encoder barely cares; a degraded/low-precision one drops off fast. Run a
probe at level 0 vs level 2 and compare the accuracy - that gap is the signal.

Everything works on the base64 data-URL strings evalscope puts in image content,
so it slots straight into a loaded dataset.
"""
from __future__ import annotations

import base64
import io


def _decode(data_url: str):
    from PIL import Image
    raw = data_url.split(',', 1)[1] if data_url.startswith('data:') else data_url
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert('RGB')


def _encode(img, fmt='PNG') -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return f'data:image/{fmt.lower()};base64,' + base64.b64encode(buf.getvalue()).decode()


def downscale_upscale(img, factor: float):
    """Shrink then blow back up - throws away high-frequency detail."""
    w, h = img.size
    from PIL import Image
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def jpeg(img, quality: int):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf).convert('RGB')


def blur(img, radius: float):
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius))


def quantize_colors(img, n_colors: int):
    return img.quantize(colors=n_colors).convert('RGB')


# level -> (downscale factor, jpeg quality, blur radius, colours). Level 0 = untouched.
_LADDER = {
    1: (0.75, 75, 0.5, 64),
    2: (0.50, 50, 1.0, 32),
    3: (0.33, 30, 1.5, 16),
}


def degrade_data_url(data_url: str, level: int) -> str:
    """Apply the level-`level` stress to one image given as a data URL."""
    if level <= 0:
        return data_url
    factor, q, radius, colors = _LADDER.get(level, _LADDER[3])
    try:
        img = _decode(data_url)
    except Exception:
        return data_url  # if we can't read it, leave it alone
    img = downscale_upscale(img, factor)
    img = jpeg(img, q)
    img = blur(img, radius)
    img = quantize_colors(img, colors)
    return _encode(img)
