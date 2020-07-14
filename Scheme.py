import math
import numpy as np
from PIL import Image


def colordif(pix1, pix2):
    c = [abs(pix1[0] - pix2[0]),
         abs(pix1[1] - pix2[1]),
         abs(pix1[2] - pix2[2])]
    return round(sum(c) / 3)


def getavg(data):
    avg = np.round(np.average(data, axis=0)).astype(np.int)
    return avg.tolist()


def tohex(col):
    string = '#'
    for num in col:
        h = hex(num)[2:]
        if len(h) == 1:
            h = '0' + h
        string += h
    return string.upper()


def binpixels(img, coords, rad):
    sy = coords[0]
    sx = coords[1]
    w = rad + 1
    pxls = []
    for y in range(w):
        for x in range(w):
            try:
                px = img.getpixel((sx + x, sy + y))
                pxls.append([px[0], px[1], px[2]])
            except IndexError:
                pass
    return getavg(pxls)


def binimage(img, rad):
    w = math.floor(img.width / (rad + 1))
    print(img.width, w, w * (rad + 1))
    h = math.floor(img.height / (rad + 1))
    binned = Image.new('RGB', (w, h), 'white')
    print('Initiating array...')
    pxmp = np.array([[[0, 0, 0]] * w] * h, dtype=np.int8)
    print('Populating array...')
    for y in range(h):
        for x in range(w):
            px = binpixels(img, (y * (rad + 1), x * (rad + 1)), rad)
            pxmp[y][x] = px
            binned.putpixel((x, y), (px[0], px[1], px[2]))
        print(y, h)
    binned.save('Binned.png')
    return pxmp


def preparr(arr):
    print('Flattening array...')
    px = np.reshape(arr, (count, 3))
    px = px.astype(np.int16)
    print('Ensuring no negative values...')
    for p in px:
        for x in range(3):
            p[x] = abs(p[x])
    return px


def groupx(pxls, thresh):
    print('Initiating index array...')
    added = [False] * count
    print('Starting grouping process...')
    grps = []
    while False in added:
        p = added.index(False)
        px = pxls[p]
        grps.append([])
        x = 0
        while len(pxls) > x:
            p = pxls[x]
            if not added[x]:
                dif = colordif(px, p)
                if dif < thresh:
                    grps[-1].append(p.tolist())
                    added[x] = True
            x += 1
        print(added.count(True))
    return grps


def merge(grps):
    print('Merging groups...')
    rgb = []
    hx = []
    for grp in grps:
        c = getavg(grp)
        rgb.append(c)
        hx.append(tohex(c))
    return rgb, hx


def output(rgb, hx):
    with open('Pallete.txt', 'w') as out:
        lines = ['From ' + file + ':\n', '\n']
        lines.extend([c + '\n' for c in hx])
        out.writelines(lines)
    out = Image.new('RGB', (50 * len(rgbs), 100), 'black')
    for c in range(len(rgb)):
        for x in range(50):
            for y in range(100):
                out.putpixel((x + (50 * c), y), (rgb[c][0], rgb[c][1], rgb[c][2]))
    out.save('Palette.png')


path = '/usr/share/backgrounds/'
file = 'Sand Dune.png'

print('Opening image...')
image = Image.open(path + file)
count = image.width * image.height
print(str(count) + ' pixels.')

pixmap = binimage(image, 3)

count = pixmap.shape[0] * pixmap.shape[1]

pixels = preparr(pixmap)

groups = groupx(pixels, 24)

rgbs, hexs = merge(groups)

output(rgbs, hexs)
