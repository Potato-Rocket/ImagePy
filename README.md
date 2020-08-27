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

### Setup
The config file (named 'config.ini') should be in the same directory as the program. You should have a sample config file which you downloaded along with the program, which contains only the most basic sections. It should be commented enough to explain how to use it. It should work just fine without any modification, however It is recommended to set a default image path at the very least for convenience.

### Running
When you run this program from your Linux terminalu sing python, you can add these arguments to give your inputs more efficiently.
- "-i" or "--image" [image name] for the input image's name.
- "-f" or "--full" to make the image input specify the full directory.
- "-v" or "--verbose" for verbose output.