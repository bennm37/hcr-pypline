import os


def get_path(name="dropbox.txt"):
    if os.path.exists(f"data/{name}"):
        with open(f"data/{name}", "r") as f:
            return f.read().strip()
    else:
        raise FileNotFoundError
