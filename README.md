# ImagePy
A simple tool written in python to get a color scheme from an image.

This project is still in its early develompent stages, so better documentation will be created later. However, the code (Scheme.py) is heavily commented if you want to check it out and give suggestions.

The goal is to provide a tool that creates a color scheme that reflects all the colors in an image by thoroughly analyzing it. It will also be able to write directly to a user defined set of config files, eg. a terminal or WM config file.

### How It Works
It gets the color palette in five separate phases:
1. Preparation: Quickly scaling the image down to be more manageable if it is above a certain size, then putting the pixels into an array which stores their RGB values.
2. Binning: This goes through the image looking at one chunk at a time, and returning the average of the chunk. If the variance of the pixels in a chunk is above a certain threshold, it returns the value as black. This new pixel is added to a new array of pixels.
3. Flattening: Prepares the pixels for grouping by removing pixels under a certain level of brightness or saturation (removing the black pixels) and converting the 2D array to a 1D list. It then combines pixels with identical color values and adds a count attribute to speed up the grouping algorithm.
4. Grouping: Groups pixels that are within a certain range from a starter pixel together and adds that group to a seperete list. It repeats this until all pixels have been grouped.
5. Refining: Repeats the Grouping process, altering the threshold until there are the desired number of colors. It then tries to sort the colors in a preferable order.

### Sample Output

Source image (*Howl's Flying Castle*):

![Moving Castle](sample/Moving%20Castle.jpg)

Binned image (intermediate step):

![Binned](sample/Binned.png)

Generated palette:

![Palette](sample/Palette.png)

```
#D7E7BC  #AFCB8C  #1D2E30  #537D43
#4AC09C  #83D1A1  #7B9E56  #AE5535
```

### Setup
Install dependencies with [uv](https://docs.astral.sh/uv/):
```
uv sync
```

The config file (`config.ini`) should be in the same directory as the program. A `sample-config.ini` is included — copy it to `config.ini` and edit from there. At minimum, set the `images` path under `[Paths]` to your wallpaper directory. The sample config is commented throughout, including an example `[user/myapp]` section for writing colors to other config files.

### Running
```
uv run python Scheme.py -i "image name.jpg"
```

Arguments:
- `-i` / `--image` — image filename (relative to the `images` path in config)
- `-f` / `--full` — treat the `-i` value as a full path instead
- `-v` / `--verbose` — verbose output