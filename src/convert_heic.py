import glob
import os
import pillow_heif
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()

for path in glob.glob("data/heic/*.HEIC"):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    name = os.path.splitext(os.path.basename(path))[0] + ".jpg"
    img.save(os.path.join("data/raw", name), quality=95)
    print("Converted", name)