import os
import sys
import csv
import math
import colorsys
import configparser as cp
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing import Process, cpu_count


class GetColors:
    # Initiates the class to get a color palette
    def __init__(self, img, pxbin, thresh, imgsize, size, value):
        print('Opening image...')
        self.binning = pxbin
        self.threshold = thresh
        self.file = img
        self.value = value
        self.palettesize = size
        # Open the image and get info about the image
        try:
            self.image = Image.open(self.file)
        except FileNotFoundError:
            print('\nFileNotFoundError: No such file or directory: \'' + self.file + '\'')
            sys.exit(2)
        width, height = self.image.size
        if width > imgsize:
            print('Downsizing image...')
            ratio = imgsize / width
            self.image = self.image.resize((imgsize, int(height * ratio)))
        self.count = self.image.width * self.image.height
        print(str(self.count) + ' pixels.')
        self.cores = cpu_count()

    # Runs all functions required to get the color pallete
    def run(self):
        threads = []
        chunk = int(math.floor(self.image.height / self.cores))
        chunks = [self.image.crop((0, t * chunk, self.image.width, (t + 1) * chunk)) for t in range(self.cores)]

        for t in range(self.cores):
            p = Process(target=self.binimage, args=(chunks[t], self.binning, t,))
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

        pixels = self.preparr(pixmap, self.value)
        while True:
            print('Threshold: ' + str(self.threshold))
            colors = self.groupx(pixels, self.threshold)
            length = len(colors)
            step = abs(length - self.palettesize)
            if length == self.palettesize:
                break
            elif length < self.palettesize:
                self.threshold -= step
            else:
                self.threshold += step
        colors = self.sortcols(colors)
        self.output(colors)

    # Get HSV value from RGB concisely
    @staticmethod
    def gethsv(rgb):
        hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
        return hsv

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
    def sortcols(self, rgb):
        # Sort by converting to hsv
        sort = sorted(rgb, key=lambda x: (self.gethsv(x)[0], self.gethsv(x)[2], self.gethsv(x)[1]))
        # Separate and return the colors and lengths
        return sort

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
        else:
            px = self.getavg(pxls)
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
        print(str(len(px)) + ' unique colors to group.')
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
            # Get the most common pixel in the list of pixels
            px = pix[np.where(pix == np.max(pix[:, 3]))[0][0]]
            # Get the differences between the first pixel and the remaining pixels
            difs = np.apply_along_axis(self.colordif, 1, pix, px)
            # Get the indexes of pixels that are within the threshold
            ind = np.where(difs < thresh)
            # Get the list of pixels within the threshold
            grp = pix[ind].tolist()
            grps.append(grp)
            # Remove the pixels that were added to the group from the remaining pixels
            pix = np.delete(pix, ind, axis=0)
        colors = self.merge(grps)
        return colors

    # Get the average color for each group
    def merge(self, grps):
        gps = grps.copy()
        print('Merging groups...')
        # Start color lists
        col = []
        # For each group
        for grp in gps:
            full = []
            for px in grp:
                val = self.gethsv(px)[2]
                for _ in range(val):
                    full.append(px[:3])
            # Add the average color for the group to the rgb and hex lists
            c = self.getavg(grp)
            col.append(c)
        # Return the lists of color values
        return col

    # Output the generated list of colors
    def output(self, rgb):
        # Open palette.txt to write
        with open('Palette.txt', 'w') as out:
            # Create a list of lines starting with info about the base image
            lines = ['From ' + self.file + ':\n', '\n']
            # Add the hex color values to the list of lines and write the lines
            lines.extend([self.tohex(c) + '\n' for c in rgb])
            out.writelines(lines)
        # Create a new image to display the color palette
        out = Image.new('RGB', (50 * len(rgb), 100), 'black')
        for c in range(len(rgb)):
            for x in range(50):
                for y in range(100):
                    # Fill in the current color
                    out.putpixel((x + (50 * c), y), (rgb[c][0], rgb[c][1], rgb[c][2]))
        out.save('Palette.png')


class Config:
    # Initiates the class to get the data from a config file
    def __init__(self, directory):
        self.config = cp.ConfigParser()
        self.file = directory

    def read(self):
        self.config.read(self.file)
        dflt = {'palette-size': int(self.getkey('Defaults', 'palette-size', 8)),
                'color-value-limit': int(self.getkey('Defaults', 'color-value-limit', 16))}

        pth = {'images': self.getkey('Paths', 'images', '/usr/share/backgrounds/')}

        alg = {'start-threshold': int(self.getkey('Algorithm', 'start-threshold', 64)),
               'binning-size': int(self.getkey('Algorithm', 'binning-size', 9)),
               'binning-variance-limit': int(self.getkey('Algorithm', 'binning-variance-limit', 32)),
               'image-resize-limit': int(self.getkey('Algorithm', 'image-resize-limit', 1920))}

        return dflt, pth, alg

    def getkey(self, section, key, default):
        try:
            value = self.config[section][key]
        except KeyError:
            value = default
        return value


# Choose an image
image = 'Quasar.png'

config = Config('config.ini')
defaults, paths, algorithm = config.read()
imgcolor = GetColors(paths['images'] + image,
                     algorithm['binning-size'],
                     algorithm['start-threshold'],
                     algorithm['image-resize-limit'],
                     defaults['palette-size'],
                     defaults['color-value-limit'])
imgcolor.run()
