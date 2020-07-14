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
    pxmp = np.array([[[0, 0, 0]] * img.width] * img.height, dtype=np.int8)
    print('Populating array...')
    for y in range(img.height):
        print(str(y * img.width) + '/' + str(count))
        for x in range(img.width):
            px = img.getpixel((x, y))
            for p in range(3):
                pxmp[y][x][p] = px[p]
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
        print(added.count(True))
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
    return grps


def merge(grps):
    print('Merging groups...')
    rgb = []
    hx = []
    for grp in grps:
        array = np.add.reduce(np.array(grp))
        c = [int(round(array[0] / len(grp))),
             int(round(array[1] / len(grp))),
             int(round(array[2] / len(grp)))]
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


path = '/home/oscar/Pictures/Gimp/Exports/'
file = 'Test Image.png'

print('Opening image...')
image = Image.open(path + file)
count = image.width * image.height
print(str(count) + ' pixels.')

pixmap = binimage(image)

pixels = preparr(pixmap)

groups = groupx(pixels, 16)

rgbs, hexs = merge(groups)

output(rgbs, hexs)
