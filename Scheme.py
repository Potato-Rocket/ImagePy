#! /usr/bin/python3

import os
import sys
import csv
import math
import getopt
import colorsys
import configparser
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing import Process, cpu_count


def verbose(string):
    # Only prints a string if verbose output is enabled
    if ver:
        print(string)


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
            verbose('Downsizing image...')
            ratio = imgsize / width
            self.image = self.image.resize((imgsize, int(height * ratio)))
        self.count = self.image.width * self.image.height
        verbose(str(self.count) + ' pixels.')
        self.cores = cpu_count()

    # Runs all functions required to get the color pallete
    def run(self):
        print('Starting binning process...')

        threads = []
        chunk = int(math.floor(self.image.height / self.cores))
        chunks = [self.image.crop((0, t * chunk, self.image.width, (t + 1) * chunk)) for t in range(self.cores)]

        for t in range(self.cores):
            p = Process(target=self.binimage, args=(chunks[t], self.binning, t,))
            threads.append(p)
            p.start()
        for t in threads:
            t.join()

        verbose('Combining threads...')
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

        verbose('Building binned image...')
        binned = Image.new('RGB', (width, height))
        for y in range(height):
            for x in range(width):
                px = pixmap[y][x]
                binned.putpixel((x, y), (px[0], px[1], px[2]))
        binned.save('Binned.png')

        self.count = pixmap.shape[0] * pixmap.shape[1]

        pixels = self.preparr(pixmap, self.value)
        print('Starting grouping process...')
        while True:
            verbose('Threshold: ' + str(self.threshold))
            colors = self.groupx(pixels, self.threshold)
            length = len(colors)
            step = abs(length - self.palettesize)
            if length == self.palettesize:
                break
            elif length < self.palettesize:
                self.threshold -= step
            else:
                self.threshold += step
        print('Merging groups...')
        colors = self.sortcols(colors)
        self.output(colors)

    # Get HSV value from RGB concisely
    @staticmethod
    def gethsv(rgb):
        raw = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
        hsv = [int(raw[0] * 256), int(raw[1] * 256), int(raw[2])]
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
        # Sort by the hue and saturation, combined through multiplication, to get color intensity
        intensity = [list(i) for i in sorted(rgb, key=lambda x: self.gethsv(x)[1] * self.gethsv(x)[2])]
        # Get half the length of the color palette
        length = int(math.floor(len(intensity) / 2))
        # Sort the top half of colors by hue, with reds at the top
        top = sorted(intensity[length:], key=lambda x: self.gethsv(x)[0], reverse=True)
        # Sort the bottom half of colors by saturation
        bottom = sorted(intensity[:len(intensity) - length], key=lambda x: self.gethsv(x)[1])
        # Combine the sorted top and bottom halves and return the colors
        sort = bottom + top
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
        verbose('Started binning thread ' + str(pid + 1))
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
            verbose(str(pid + 1) + ': ' + str(int((100 / h) * y)) + '%')
        verbose('Completed thread ' + str(pid + 1))
        with open('img' + str(pid) + '.csv', 'w') as out:
            writer = csv.writer(out)
            shp = np.shape(pxmp)
            writer.writerows(pxmp.reshape(shp[0], shp[1] * 3))

    # Prepare pixel list for the grouping algorithm
    def preparr(self, arr, dark):
        verbose('Flattening array...')
        # Reshape the 2D pixel array to a 1D pixel array
        px = np.reshape(arr, (self.count, 3))

        data = pd.DataFrame(np.append(px, np.ones((len(px), 1)), axis=1),
                            columns=['R', 'G', 'B', 'Count'], dtype=np.long)
        data = data.groupby(['R', 'G', 'B']).agg({'R': 'first',
                                                  'G': 'first',
                                                  'B': 'first',
                                                  'Count': 'sum'}).reset_index(drop=True)
        px = np.array(data)
        # Removes all colors darker than the limit
        hsv = np.apply_along_axis(self.gethsv, 1, px)
        px = px[np.where(np.logical_or(hsv[:, 1] >= dark, hsv[:, 2] >= dark))]

        verbose(str(len(px)) + ' unique colors to group.')
        # Return the new array
        return px

    # Separate pixel list into color groups
    def groupx(self, pxls, thresh):
        # Copy the list of pixels so it can be modified
        pix = np.copy(pxls)
        full = len(pix)
        # Create the list of groups
        grps = []
        # Repeat while there are still pixels in the list
        while len(pix > 0):
            # Print the percent remaining
            verbose(str(100 - int((100 / full) * len(pix))) + '%')
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
        self.config = configparser.ConfigParser()
        self.file = directory

    # Reads the config files, and adds/overwrites values based on command line inputs
    def read(self):
        # Read the basic keys from the config file
        dflt, pth, alg = self.getkeys()

        # Get the command line arguments and make sure they are correct
        args = sys.argv[1:]
        short = 'i:fv'
        long = ['image=', 'full', 'verbose']
        try:
            arguments, values = getopt.getopt(args, short, long)
        except getopt.error as error:
            print(str(error))
            sys.exit(2)

        # Get the inputs and set valuse from the command line arguments
        dflt['verbose'] = False
        for arg, val in arguments:
            if arg in ('-i', '--image'):
                pth['image'] = val
            if arg in ('-f', '--full'):
                pth['images'] = ''
            if arg in ('-v', '--verbose'):
                dflt['verbose'] = True

        # Return the keys, which may have been overridden by the command line inputs
        return dflt, pth, alg

    # Reads the config file to get all necessary alues, and returns dictionaries
    def getkeys(self):
        # Read the config file
        self.config.read(self.file)
        # Get all the keys from the default section and add them to a dictionary
        dflt = {'palette-size': self.getkey('Defaults', 'palette-size', 8, 'int'),
                'color-value-limit': self.getkey('Defaults', 'color-value-limit', 16, 'int'),
                'config-files': self.getkey('Defaults', 'config-files', '', 'string').split()}

        # Get the path from the paths section, and add it to a dictionary
        pth = {'images': self.getkey('Paths', 'images', '', 'string')}

        # Get the keys from the algorithm section, and add them to a dictionary
        alg = {'start-threshold': self.getkey('Algorithm', 'start-threshold', 64, 'int'),
               'binning-size': self.getkey('Algorithm', 'binning-size', 9, 'int'),
               'binning-variance-limit': self.getkey('Algorithm', 'binning-variance-limit', 32, 'int'),
               'image-resize-limit': self.getkey('Algorithm', 'image-resize-limit', 1920, 'int')}
        # Return all three dictionaries
        return dflt, pth, alg

    # Reads values necessary to write the wallpaper
    def getwalkeys(self):
        # Get all the keys from the wallpaper section and add them to a dictionary
        wal = {'file': self.getkey('Wallpaper', 'file', '', 'string'),
               'comment': self.getkey('Wallpaper', 'comment', '', 'string'),
               'line': self.getkey('Wallpaper', 'line', '', 'string'),
               'set': self.getkey('Wallpaper', 'set', False, 'bool'),
               'set-immediately': self.getkey('Wallpaper', 'set-immediately', False, 'bool'),
               'command': self.getkey('Wallpaper', 'command', '', 'string')}
        # Return the dictionary
        return wal

    # Reads the values from a custom section necessary to write the palette to a config file
    def getcustomsection(self, section):
        # Get the keys from the section, and add them too a dictionary
        array = [self.getkey(section, 'file', '', 'string'),
                 self.getkey(section, 'start-comment', '', 'string'),
                 self.getkey(section, 'end-comment', '', 'string'),
                 self.getkey(section, 'line', '', 'string'),
                 self.getkey(section, 'colors', '', 'string'),
                 self.getkey(section, 'numbers', '', 'string')]
        # If a key returns empty, return nothing
        if '' in array:
            return {}
        else:
            return {'file': array[0], 'start-comment': array[1], 'end-comment': array[2],
                    'line': array[3], 'colors': array[4], 'numbers': array[5]}

    # Gets a certain key with  a certain data type from the config file
    # If it hits an error, it returns the default value
    def getkey(self, section, key, default, outype):
        if outype == 'bool':
            try:
                value = self.config.getboolean(section, key)
            except ValueError:
                value = default
        else:
            try:
                value = self.config[section][key]
            except KeyError:
                value = default
            if outype == 'int':
                try:
                    value = int(value)
                except ValueError:
                    value = default
        return value


class Write:
    # Initiates class which handles writing color palette
    def __init__(self):
        pass

    # Sets the desktop wallpeper based on the source image
    @staticmethod
    def wallpaper(var, img):
        # Checks if it should update the config file
        if var['set']:
            print('Setting wallpaper...')
            # Generste the line to set the wallpaper
            line = var['line']
            line = line.replace('%B', img).strip('\'') + '\n'
            # Try to open the config file
            # If sucessful keep going
            try:
                with open(var['file'], 'r') as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print('Error: No such file or directory: ' + var['file'])
            else:
                # Try to find the comment
                # If succesful keep going
                try:
                    index = lines.index(var['comment'].strip('\'') + '\n') + 1
                except ValueError:
                    print('Error: Specified comment not found in specified file.')
                else:
                    # Replace the line after the comment with the generated line, and write the file
                    lines[index] = line
                    verbose(var['file'])
                    with open(var['file'], 'w') as file:
                        file.writelines(lines)

        if var['set-immediately']:
            print('Updating wallpaper...')
            string = var['command']
            string = string.replace('%B', '\'' + img + '\'')
            verbose(string)
            os.system(string)

    # Writes to each config file as specified by custom sections
    def colorpalette(self, cfg, files):
        print('Updating color scheme...')
        # Get the color pallete
        with open('Palette.txt', 'r') as file:
            colors = file.readlines()[2:]
        colors = [col.strip('\n') for col in colors]
        # For each config file
        for conf in files:
            # Try to get the keys from the custom section
            args = cfg.getcustomsection('user/' + conf)
            if args != {}:
                verbose(args['file'])
                # Try to open the config file
                # If successful, proceed
                try:
                    with open(args['file'], 'r') as file:
                        lines = file.readlines()
                except FileNotFoundError:
                    print('Error: ' + args['file'] + ' does not exist.')
                else:
                    # Split the lines into chunks based on whene the start and end comments are
                    startind = lines.index(args['start-comment'].strip('\'').strip('\"') + '\n')
                    endind = lines.index(args['end-comment'].strip('\'').strip('\"') + '\n')
                    startlines = lines[:startind + 1]
                    endlines = lines[endind:]
                    # Get which colors to use and how to assign them
                    inds = self.getindexes(args['colors'])
                    cols = [colors[x] for x in inds]
                    nums = self.getindexes(args['numbers'])
                    # Generate the lines to set the colors
                    midlines = []
                    for x in range(len(nums)):
                        ind = x % len(cols)
                        line = args['line']
                        midlines.append(line.replace('%C', cols[ind]).replace('%N', str(nums[x])) + '\n')
                    # Combine the chunks and insert the lines to set the colors, then write the file
                    lines = startlines
                    lines.extend(midlines)
                    lines.extend(endlines)
                    with open(args['file'], 'w') as file:
                        file.writelines(lines)

    # Interpret the index string to return a list of indexes
    # Adds together all index ranges, which are separated by spaces
    # Ranges can be a single index, or a range of indexes, defined by two indexes separated by a colon
    @staticmethod
    def getindexes(string):
        inds = string.split()
        out = []
        for col in inds:
            if ':' in col:
                rng = col.split(':')
                try:
                    rng[0] = int(rng[0])
                    rng[1] = int(rng[1])
                except ValueError:
                    pass
                else:
                    for x in range(rng[0], rng[1]):
                        out.append(x)
            else:
                try:
                    out.append(int(col))
                except ValueError:
                    pass
        return out


config = Config('config.ini')
defaults, paths, algorithm = config.read()
ver = defaults['verbose']
try:
    print(paths['image'])
except KeyError:
    paths['image'] = input('Image directory: ')

image = paths['images'] + paths['image']

imgcolor = GetColors(image,
                     algorithm['binning-size'],
                     algorithm['start-threshold'],
                     algorithm['image-resize-limit'],
                     defaults['palette-size'],
                     defaults['color-value-limit'])
imgcolor.run()

wallpaper = config.getwalkeys()
write = Write()
write.wallpaper(wallpaper, image)
write.colorpalette(config, defaults['config-files'])
