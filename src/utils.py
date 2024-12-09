import os
import glob

def get_image_paths(input_dir, extensions = ("jpg", "jpeg", "png", "bmp")):
    """
    找到所有图片文件路径
    """
    # 输入是图片文件路径
    if os.path.isfile(input_dir):
        assert any(input_dir.lower().endswith(extension) for extension in extensions)
        return [input_dir]

    # 输入是文件夹路径
    pattern = f"{input_dir}/**/*"
    img_paths = []

    for extension in extensions:
        img_paths.extend(glob.glob(f"{pattern}.{extension}", recursive=True))

    if not img_paths:
        raise FileNotFoundError(f"No images found in {input_dir}. Supported formats are: {', '.join(extensions)}")

    return img_paths