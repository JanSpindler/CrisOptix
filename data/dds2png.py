from PIL import DdsImagePlugin, Image
import os
import tqdm


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)

    for subdir, dirs, files in os.walk(dir_path):
        for file in tqdm.tqdm(files):
            if file.endswith('.dds'):
                file_path = os.path.join(subdir, file)
                with Image.open(file_path) as image:
                    image.load()
                    image.save(file_path.replace('.dds', '.png'), 'PNG')
