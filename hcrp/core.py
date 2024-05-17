import os


def get_data_path():
    if os.path.exists("data/image_location.txt"):
        with open("data/image_location.txt", "r") as f:
            return f.read().strip()
