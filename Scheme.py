import math
import random
import numpy as np
from PIL import Image


# Get the value of a color
def colorval(pix):
    avg = int(sum(pix[0:3]))
    return avg


# Get diference between two RGB values
def colordif(pix1, pix2):
    # Get the difference between each two pixel's color value
    c = [abs(pix1[0] - pix2[0]),  # Red
         abs(pix1[1] - pix2[1]),  # Green
         abs(pix1[2] - pix2[2])]  # Blue
    # Return the squared average difference
    return round(sum(c) / 3) ** 2


# Returns the average color of a set of colors, using NumPy
def getavg(data):
    # Get the NumPy rounded average
    avg = np.round(np.average(data, axis=0))
    # Return as an int
    return avg.astype(np.int64)


# Converts an RGB value to a hex string
def tohex(col):
    # Create string
    string = '#'
    # For each color value
    for num in range(3):
        # Get the numbers in the hex string for the value
        h = hex(col[num])[2:]
        # Makes sure the sub-string is two characters
        if len(h) == 1:
            h = '0' + h
        # Appends the sub-string to the main  string
        string += h
    # Returns the string in all caps
    return string.upper()


# Gets value for a pixel based on surrounding pixels
def binpixels(img, coords, rad):
    # Set the root pixel's coords
    sy = coords[0]
    sx = coords[1]
    # Get how many vertical and horizontal pixels to look at
    w = rad + 1
    # Init pixels list and iterate over x and y for the width
    pxls = []
    for y in range(w):
        for x in range(w):
            # Try to add the pixel corresponding to the current x and y to the pixel list
            # If the pixel doesn't exist, keep going
            try:
                px = img.getpixel((sx + x, sy + y))
                pxls.append([px[0], px[1], px[2]])
            except IndexError:
                pass
    # Return the average color of the list of pixels
    return getavg(pxls)


# Move image pixels to an array, and scale down using the average for chunks of pixels
def binimage(img, rad):
    # Get the new width and height based on the chunk size
    w = math.floor(img.width / (rad + 1))
    h = math.floor(img.height / (rad + 1))
    # Create a new image to show the result of this function
    binned = Image.new('RGB', (w, h), 'white')
    print('Initiating array...')
    # Create an array for the pixels
    pxmp = np.array([[[0, 0, 0, 0]] * w] * h)
    print('Populating array...')
    # Iterate through x and y, for the predetermined width and height
    for y in range(h):
        for x in range(w):
            # Get the average value for this pixel and surrounding pixels
            px = binpixels(img, (y * (rad + 1), x * (rad + 1)), rad)
            px = np.append(px, 0)
            # Add it to the list of pixels and the new image
            pxmp[y][x] = px
            binned.putpixel((x, y), (px[0], px[1], px[2]))
        print(h - y)
    # Save the downsixed image for viewing and return the list of pixels
    binned.save('Binned.png')
    return pxmp


# Prepare pixel list for the grouping algorithm
def preparr(arr):
    print('Flattening array...')
    # Reshape the 2D pixel array to a 1D pixel array
    px = np.reshape(arr, (count, 4))
    # Return the new array
    return px


# Separate pixel list into color groups
def groupx(pxls, thresh):
    # Copy the list of pixels so it can be modified
    pix = np.copy(pxls)
    print('Starting grouping process...')
    # Create the list of groups
    grps = []
    # Repeat while there are still pixels in the list
    while len(pix > 0):
        print(len(pix))
        # Get the first pixel in the list of pixels
        px = pix[0]
        # Create new group in list of groups
        grps.append([])
        x = 0
        # Repeat for every pixel in the list of pixels
        while len(pix) > x:
            # Get the color value for the pixel
            p = pix[x]
            # Compare it to the first pixel in this group
            dif = colordif(px, p)
            # If the difference is within the threshold, add it to the group
            if dif < thresh:
                grps[-1].append(p.tolist())
                pix = np.delete(pix, x, axis=0)
            else:
                x += 1
    # Return the list of groups
    return grps


# Get the average color for each group
def merge(grps):
    print('Merging groups...')
    # Start rgb and hex color lists
    rgb = []
    hx = []
    # For each group
    for grp in grps:
        print(len(grp))
        # If there are more then a certain number of pixels in the group
        if len(grp) > 32:
            # Add the average color for the group to the rgb and hex lists
            c = getavg(grp)
            rgb.append(c)
            hx.append(tohex(c))
    # Return the lists of color values
    return rgb, hx


# Output the generated list of colors
def output(rgb, hx):
    # Open Pallete.txt to write
    with open('Pallete.txt', 'w') as out:
        # Create a list of lines starting with info about the base image
        lines = ['From ' + file + ':\n', '\n']
        # Add the hex color values to the list of lines and write the lines
        lines.extend([c + '\n' for c in hx])
        out.writelines(lines)
    # Create a new image to display the color pallete
    out = Image.new('RGB', (50 * len(rgbs), 100), 'black')
    # For each color, iterate over a 50x100 block
    for c in range(len(rgb)):
        for x in range(50):
            for y in range(100):
                # Fill in the current color
                out.putpixel((x + (50 * c), y), (rgb[c][0], rgb[c][1], rgb[c][2]))
    out.save('Palette.png')


# Choose an image and image directory
# path = '/usr/share/backgrounds/'
path = '/home/oscar/Pictures/Gimp/Exports/'
file = 'Mechanized Metropolis.png'
# file = 'Test Image.png'

print('Opening image...')
# Open the image and get info about the image
image = Image.open(path + file)
count = image.width * image.height
print(str(count) + ' pixels.')

pixmap = binimage(image, 2)

count = pixmap.shape[0] * pixmap.shape[1]

pixels = preparr(pixmap)

groups = groupx(pixels, 50)

rgbs, hexs = merge(groups)

output(rgbs, hexs)
