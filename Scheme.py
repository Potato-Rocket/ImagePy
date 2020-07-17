import os
import csv
import math
import colorsys
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing import Process, cpu_count


class GetColors:
    # Initiates the class to get a color palette
    def __init__(self, img, pxbin, thresh, size):
        print('Opening image...')
        self.binning = pxbin
        self.threshold = thresh
        # Open the image and get info about the image
        self.image = Image.open(img)
        width, height = self.image.size
        if width > size:
            print('Downsizing image...')
            ratio = size / width
            self.image = self.image.resize((size, int(height * ratio)))
        self.image.save('Downsized.png')
        self.count = self.image.width * self.image.height
        self.cores = cpu_count()

    # Runs all functions required to get the color pallete
    def run(self):
        threads = []
        chunk = int(math.floor(self.image.height / self.cores))
        chunks = [self.image.crop((0, t * chunk, self.image.width, (t + 1) * chunk)) for t in range(self.cores)]

        for t in range(self.cores):
            p = Process(target=self.binimage, args=(chunks[t], self.binning, t, ))
            threads.append(p)
            p.start()
        for t in threads:
            t.join()

        print('Combining threads...')
        pixmap = np.array([])
        width = 0
        height = 0
        for r in range(self.cores):
            with open('img' + str(r) + '.csv') as raw:
                reader = csv.reader(raw)
                for row in reader:
                    width = int(len(row) / 3)
                    height += 1
                    pixmap = np.append(pixmap, np.array(row))
            os.remove('img' + str(r) + '.csv')
        pixmap = pixmap.reshape((height, width, 3)).astype(np.int64)

        print('Building binned image...')
        binned = Image.new('RGB', (width, height))
        for y in range(height):
            for x in range(width):
                px = pixmap[y][x]
                binned.putpixel((x, y), (px[0], px[1], px[2]))
        binned.save('Binned.png')

        self.count = pixmap.shape[0] * pixmap.shape[1]

        pixels = self.preparr(pixmap, 16)
        groups = self.groupx(pixels, threshold)
        rgbs, lens = self.merge(groups)
        rgbs, lens = self.sortcols(rgbs, lens)
        self.output(rgbs, lens)

    # Returns the average color of a set of colors, using NumPy
    @staticmethod
    def getavg(data):
        # Get the NumPy rounded average
        avg = np.round(np.average(data, axis=0))
        # Return as an int
        return avg.astype(np.int64)

    # Get diference between two RGB values
    @staticmethod
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

    # Converts an RGB value to a hex string
    @staticmethod
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

    # Sort the color pallete by hue, saturation, or value
    @staticmethod
    def sortcols(rgb, lw):
        # Combine the lengths and colors so they get sorted together
        lws = np.array(lw)
        lws = np.reshape(lws, (len(lws), 1))
        cols = np.append(rgb, lws, axis=1)
        # Sort by converting to hsv
        sort = sorted(cols, key=lambda x: colorsys.rgb_to_hsv(x[0], x[1], x[2])[0])
        # Separate and return the colors and lengths
        cols = [x[0:3] for x in sort]
        lws = [x[3] for x in sort]
        return cols, lws

    # Get variance of a set of pixels
    def getvar(self, data):
        dat = np.array(data)
        mean = self.getavg(dat)
        if len(np.shape(dat)) == 2:
            difs = np.apply_along_axis(self.colordif, 1, dat, mean) ** 2
        else:
            difs = abs(dat - mean) ** 2
        variance = math.sqrt(sum(difs) / len(difs))
        return int(variance)

    # Gets value for a pixel based on surrounding pixels
    def binpixels(self, img, coords, rad):
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
        variance = self.getvar(pxls)
        if variance > 32:
            px = np.array([0, 0, 0])
        elif variance > 8:
            px = self.getavg(pxls)
        else:
            px = pxls[0]
        # Return the average color of the list of pixels
        return px

    # Move image pixels to an array, and scale down using the average for chunks of pixels
    def binimage(self, img, rad, pid):
        print('Started binning thread ' + str(pid + 1))
        # Get the new width and height based on the chunk size
        w = math.floor(img.width / (rad + 1))
        h = math.floor(img.height / (rad + 1))
        # Create an array for the pixels
        pxmp = np.array([[[0, 0, 0]] * w] * h)
        # Iterate through x and y, for the predetermined width and height
        for y in range(h):
            for x in range(w):
                # Get the average value for this pixel and surrounding pixels
                px = self.binpixels(img, (y * (rad + 1), x * (rad + 1)), rad)
                # Add it to the list of pixels and the new image
                pxmp[y][x] = px
            print(str(pid + 1) + ': ' + str(int((100 / h) * y)) + '%')
        print('Completed thread ' + str(pid + 1))
        with open('img' + str(pid) + '.csv', 'w') as out:
            writer = csv.writer(out)
            shp = np.shape(pxmp)
            writer.writerows(pxmp.reshape(shp[0], shp[1] * 3))

    # Prepare pixel list for the grouping algorithm
    def preparr(self, arr, dark):
        print('Flattening array...')
        # Reshape the 2D pixel array to a 1D pixel array
        px = np.reshape(arr, (self.count, 3))

        # Removes all colors darker than [16, 16, 16]
        px = px[np.logical_not(np.logical_and(px[:, 0] <= dark, px[:, 1] <= dark, px[:, 2] <= dark))]

        data = pd.DataFrame(np.append(px, np.ones((len(px), 1)), axis=1),
                            columns=['R', 'G', 'B', 'Count'], dtype=np.long)
        data = data.groupby(['R', 'G', 'B']).agg({'R': 'first',
                                                  'G': 'first',
                                                  'B': 'first',
                                                  'Count': 'sum'}).reset_index(drop=True)
        px = np.array(data)
        # Return the new array
        return px

    # Separate pixel list into color groups
    def groupx(self, pxls, thresh):
        # Copy the list of pixels so it can be modified
        pix = np.copy(pxls)
        full = len(pix)
        print('Starting grouping process...')
        # Create the list of groups
        grps = []
        # Repeat while there are still pixels in the list
        while len(pix > 0):
            # Print the percent remaining
            print(str(100 - int((100 / full) * len(pix))) + '%')
            # Get the first pixel in the list of pixels
            px = pix[0]
            # Get the differences between the first pixel and the remaining pixels
            difs = np.apply_along_axis(self.colordif, 1, pix, px)
            # Get the indexes of pixels that are within the threshold
            ind = np.where(difs < thresh)
            # Get the list of pixels within the threshold
            grp = pix[ind].tolist()
            grps.append(grp)
            # Remove the pixels that were added to the group from the remaining pixels
            pix = np.delete(pix, ind, axis=0)
        return grps

    # Get the average color for each group
    def merge(self, grps):
        gps = grps.copy()
        print('Merging groups...')
        # Startcolor lists
        col = []
        lw = []
        # Get list lengths
        for grp in gps:
            lw.append(len(grp))
        thresh = int(max(lw) / 1000)
        # Remove excessively rare color groups
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
            c = self.getavg(grp)
            col.append(c)
        # Return the lists of color values
        return col, lw

    # Output the generated list of colors
    def output(self, rgb, lw):
        # Open palette.txt to write
        with open('Palette.txt', 'w') as out:
            # Create a list of lines starting with info about the base image
            lines = ['From ' + file + ':\n', '\n']
            # Add the hex color values to the list of lines and write the lines
            lines.extend([self.tohex(c) + '\n' for c in rgb])
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
            for x in range(50):
                for y in range(100):
                    # Fill in the current color
                    out.putpixel((x + (50 * c), y), (rgb[c][0], rgb[c][1], rgb[c][2]))
        out.save('Palette.png')


# Configure basic settings

binning = 9
threshold = 32
sizelimit = 1920
prev = False

# Choose an image and image directory

# path = '/usr/share/backgrounds/'
path = '/home/oscar/Pictures/Gimp/Exports/'
# file = 'Empty Valley.png'
file = 'Test Image.png'

if prev:
    imgcolor = GetColors('Binned.png', 0, threshold, sizelimit)
else:
    imgcolor = GetColors(path + file, binning, threshold, sizelimit)

imgcolor.run()
