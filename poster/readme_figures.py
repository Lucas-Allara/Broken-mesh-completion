"""Writes the images of README.md into figures/readme.

    python poster/readme_figures.py

poster.jpg is the whole poster and hero.jpg its band with the five versions of Homer, both cut
from poster.pdf, as is social-preview.jpg, the 1280 x 640 card that GitHub shows where the
repository is linked (Settings, Social preview). results.jpg (Spot and Homer with the holes of the
22% level) and stages.jpg (the close-ups of fairing and of symmetry) are put together on white
from the figures of the poster, with its labels. pypdfium2 renders the page.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from poster_figures import _font_dir

ROOT = Path(__file__).resolve().parent.parent
POSTER = ROOT / "poster"
FIGURES = POSTER / "figures"
OUT = ROOT / "figures" / "readme"
# Boxes in millimeters of the 700 x 1000 mm page, as (left, top, right, bottom), from layout.tex:
# the labels and the image of the teaser, and the same band with the header above it.
TEASER = (0, 146, 700, 299)
HEADER_AND_TEASER = (0, 14, 700, 299)

INK, INK2 = "#2b2b2d", "#5e5e63"
METHOD, FAIRING, SYMMETRY = "#008c9a", "#b06a0f", "#820a32"


def flat(im: Image.Image) -> Image.Image:
    """The image on white, without alpha."""
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def save(im: Image.Image, name: str, width=None, quality: int = 88):
    im = flat(im)
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    path = OUT / name
    im.save(path, quality=quality, optimize=True, progressive=True)
    print(f"{path.relative_to(ROOT)}: {im.width} x {im.height}, {path.stat().st_size // 1024} KB")


def pdf_page(path: Path, width: int) -> Image.Image:
    import pypdfium2 as pdfium
    page = pdfium.PdfDocument(str(path))[0]
    return page.render(scale=width / page.get_size()[0]).to_pil()


def figure(name: str, width=None, height=None) -> Image.Image:
    """A figure of the poster on white, scaled to a width or to a height."""
    im = flat(Image.open(FIGURES / name))
    if width is None:
        width = round(im.width * height / im.height)
    return im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)


def label(draw: ImageDraw.ImageDraw, text: str, center: float, top: int, size: int, color: str) -> None:
    """A label of the poster, centered at an abscissa."""
    font = ImageFont.truetype(str(_font_dir() / "Montserrat-SemiBold.otf"), size)
    draw.text((center, top), text, font=font, fill=color, anchor="ma")


def results() -> Image.Image:
    """The rows of Spot and of Homer under the names of their five columns."""
    width, head, gap = 2000, 78, 30
    spot, homer = figure("spot_22.png", width), figure("homer_22.png", width)
    sheet = Image.new("RGB", (width, head + spot.height + gap + homer.height), "white")
    draw = ImageDraw.Draw(sheet)
    for k, text in enumerate(("Input", "Method of [1]", "With fairing", "With symmetry", "Original")):
        label(draw, text, (k + 0.5) * width / 5, 10, 40, INK)
    sheet.paste(spot, (0, head))
    # Homer stands 2.6/340 of the width to the left of the center of its cell, as blocks.tex notes.
    sheet.paste(homer, (round(width * 26 / 3400), head + spot.height + gap))
    return sheet


def stages() -> Image.Image:
    """The close-ups of fairing and of symmetry, each pair over its two labels."""
    height, foot, gap = 520, 70, 90
    cap, mirror = figure("cap.png", height=height), figure("mirror.png", height=height)
    sheet = Image.new("RGB", (cap.width + gap + mirror.width, height + foot), "white")
    draw = ImageDraw.Draw(sheet)
    sheet.paste(cap, (0, 0))
    sheet.paste(mirror, (cap.width + gap, 0))
    for text, center, color in (("method of [1]", cap.width / 4, METHOD), ("with fairing", 3 * cap.width / 4, FAIRING),
                                ("input", cap.width + gap + mirror.width / 4, INK2),
                                ("mirrored faces", cap.width + gap + 3 * mirror.width / 4, SYMMETRY)):
        label(draw, text, center, height + 12, 36, color)
    return sheet


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    poster = pdf_page(POSTER / "poster.pdf", 2800)
    s = poster.width / 700

    def cut(box):
        return poster.crop(tuple(round(v * s) for v in box))

    save(poster, "poster.jpg", 1200, 90)
    save(cut(TEASER), "hero.jpg", 1800)
    save(results(), "results.jpg", 1400)
    save(stages(), "stages.jpg", 1400)
    # The social card: the header and the teaser at 1280 px, centered on white.
    card = flat(cut(HEADER_AND_TEASER))
    card = card.resize((1280, round(card.height * 1280 / card.width)), Image.LANCZOS)
    canvas = Image.new("RGB", (1280, 640), "white")
    canvas.paste(card, (0, (640 - card.height) // 2))
    save(canvas, "social-preview.jpg", None, 90)


if __name__ == "__main__":
    main()
