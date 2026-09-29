from src.data.portable_manifest import portable_image_path


def test_portable_image_path_handles_windows_paths():
    value = r"C:\workspace\data\processed\faceforensics\real\001\000001.jpg"
    assert portable_image_path(value) == "faceforensics/real/001/000001.jpg"

