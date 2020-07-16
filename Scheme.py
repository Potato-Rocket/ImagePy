import math
import colorsys
import numpy as np
from PIL import Image


# Get variance of a set of pixels
def getvar(data):
    mean = getavg(data)
    difs = [colordif(x, mean) ** 2 for x in data]
    variance = math.sqrt(sum(difs) / len(difs))
    return int(variance)


# Get diference between two RGB values
def colordif(pix1, pix2):
    # Get the difference between each two pixel's color value
    c = [pix1[0] - pix2[0],  # Red
         pix1[1] - pix2[1],  # Green
         pix1[2] - pix2[2]]  # Blue
    c = [abs(v) ** 2 for v in c]
    dis = math.sqrt(c[0] + c[1])
    dis = math.sqrt(dis ** 2 + c[2])
    # Return the average difference
    return int(dis)


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
    variance = getvar(pxls)
    if variance > 32:
        px = np.array([0, 0, 0])
    else:
        px = getavg(pxls)
    # Return the average color of the list of pixels
    return px


# Move image pixels to an array, and scale down using the average for chunks of pixels
def binimage(img, rad):
    # Get the new width and height based on the chunk size
    w = math.floor(img.width / (rad + 1))
    h = math.floor(img.height / (rad + 1))
    # Create a new image to show the result of this function
    binned = Image.new('RGB', (w, h), 'white')
    print('Initiating array...')
    # Create an array for the pixels
    pxmp = np.array([[[0, 0, 0]] * w] * h)
    print('Populating array...')
    # Iterate through x and y, for the predetermined width and height
    for y in range(h):
        for x in range(w):
            # Get the average value for this pixel and surrounding pixels
            px = binpixels(img, (y * (rad + 1), x * (rad + 1)), rad)
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
    px = np.reshape(arr, (count, 3))
    # Removes all colors darker than [16, 16, 16]
    px = px[np.logical_not(np.logical_and(px[:, 0] < 17, px[:, 1] < 17, px[:, 2] < 17))]
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
    gps = grps.copy()
    print('Merging groups...')
    # Start rgb and hex color lists
    col = []
    lw = []
    for grp in gps:
        lw.append(len(grp))
    thresh = int(max(lw) / 1000)
    x = 0
    while x < len(lw):
        if lw[x] < thresh:
            lw.pop(x)
            gps.pop(x)
        else:
            x += 1
    # For each group
    for grp in gps:
        # Add the average color for the group to the rgb and hex lists
        c = getavg(grp)
        col.append(c)
    # Return the lists of color values
    return col, lw


# Sort the color pallete by hue, saturation, or value
def sortcols(rgb, lw):
    # Combine the lengths and colors so they get sorted together
    cols = [np.append(rgb[x], lw[x]) for x in range(len(lw))]
    # Sort by converting to hsv
    sort = sorted(cols, key=lambda x: colorsys.rgb_to_hsv(x[0], x[1], x[2])[0])
    # Separate and return the colors and lengths
    cols = [x[0:3] for x in sort]
    lws = [x[3] for x in sort]
    return cols, lws


# Output the generated list of colors
def output(rgb, lw):
    # Open palette.txt to write
    with open('Palette.txt', 'w') as out:
        # Create a list of lines starting with info about the base image
        lines = ['From ' + file + ':\n', '\n']
        # Add the hex color values to the list of lines and write the lines
        lines.extend([tohex(c) + '\n' for c in rgb])
        out.writelines(lines)
    # Create a new image to display the color palette
    out = Image.new('RGB', (100 * len(rgb), 1150), 'black')
    # Get the most common color
    most = max(lw)
    # For each color, iterate over a 50x100 block
    for c in range(len(rgb)):
        for x in range(100):
            for y in range(int(1000 * lw[c] / most)):
                # Fill in the current color based on its occurence
                out.putpixel((x + (100 * c), 999 - y), (rgb[c][0], rgb[c][1], rgb[c][2]))
        for x in range(100):
            for y in range(100):
                # Fill in the current color
                out.putpixel((x + (100 * c), 1149 - y), (rgb[c][0], rgb[c][1], rgb[c][2]))
    out.save('PaletteLarge.png')
    out = Image.new('RGB', (50 * len(rgb), 100), 'black')
    for c in range(len(rgb)):
        for x in range(100):
            for y in range(50):
                # Fill in the current color
                out.putpixel((x + (100 * c), y), (rgb[c][0], rgb[c][1], rgb[c][2]))
    out.save('Palette.png')


binning = 9
threshold = 48
prev = False

# Choose an image and image directory
path = '/usr/share/backgrounds/'
# path = '/home/oscar/Pictures/Gimp/Exports/'
file = 'SeaSunset.jpg'
# file = 'Test Image.png'

if prev:
    imgpath = './Binned.png'
    binln = 0
else:
    imgpath = path + file
    binln = binning

print('Opening image...')
# Open the image and get info about the image
image = Image.open(imgpath)
count = image.width * image.height
print(str(count) + ' pixels.')

pixmap = binimage(image, binln)
count = pixmap.shape[0] * pixmap.shape[1]
pixels = preparr(pixmap)
groups = groupx(pixels, threshold)
rgbs, lens = merge(groups)
rgbs, lens = sortcols(rgbs, lens)
output(rgbs, lens)
