import os
import numpy as np
from PIL import Image


def colordif(pix1, pix2):
    c = [abs(pix1[0] - pix2[0]),
             abs(pix1[1] - pix2[1]),
             abs(pix1[2] - pix2[2])]
    return round(sum(c) / 3)


def tohex(col):
    string = '#'
    for num in col:
        h = hex(num)[2:]
        if len(h) == 1:
            h = '0' + h
        string += h
    return string.upper()


def binimage(img):
    print('Initiating array...')
    pxmp = np.array([[[0, 0, 0]] * image.width] * image.height, dtype=np.int8)
    print('Populating array...')
    for y in range(image.height):
        print(str(y * image.width) + '/' + str(count))
        for x in range(image.width):
            px = image.getpixel((x, y))
            for p in range(3):
                pxmp[y][x][p] = px[p]
    return pxmp


os.chdir('/home/oscar/Pictures/Gimp/Exports/')
file = 'Test Image.png'

print('Opening image...')
image = Image.open(file)
count = image.width * image.height
print(str(count) + ' pixels.')

pixmap = binimage(image)

print('Flattening array...')
pixels = np.reshape(pixmap, (count, 3))
pixels = pixels.astype(np.int16)

print('Ensuring no negative values...')
for pix in pixels:
    for x in range(3):
        pix[x] = abs(pix[x])

print('Initiating index array...')
added = [False] * count

print('Starting grouping process...')
groups = []
while False in added:
    print(added.count(True))
    p = added.index(False)
    pixel = pixels[p]
    groups.append([])
    x = 0
    while len(pixels) > x:
        pix = pixels[x]
        if not added[x]:
            dif = colordif(pixel, pix)
            if dif < 16:
                groups[-1].append(pix.tolist())
                added[x] = True
        x += 1

print('Merging groups...')
colors = []
hexcol = []
for group in groups:
    array = np.add.reduce(np.array(group))
    color = [int(round(array[0] / len(group))),
             int(round(array[1] / len(group))),
             int(round(array[2] / len(group)))]
    colors.append(color)
    hexcol.append(tohex(color))

with open('Pallete.txt', 'w') as output:
    lines = ['From ' + file + ':\n', '\n']
    lines.extend([color + '\n' for color in hexcol])
    output.writelines(lines)

outimg = Image.new('RGB', (50 * len(colors), 100), 'black')
for color in range(len(colors)):
    for x in range(50):
        for y in range(100):
            outimg.putpixel((x + (50 * c), y), (colors[c][0], colors[c][1], colors[c][2]))
outimg.save('Palette.png')
