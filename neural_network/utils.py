def to_binary(img):
    threshold = 100
    transform = lambda x : 256 if x > threshold else 0
    return img.convert("L").point(transform, mode="1")

def clean_noise(img, threshold):
    pixels = img.load()

    for i in range(img.size[0]):
        pixels[i, 0] = 255
        pixels[i, img.size[1] - 1] = 255
    for i in range(img.size[1]):
        pixels[0, i] = 255
        pixels[img.size[0] - 1, i] = 255
        
    del_pos = []
    for i in range(1, img.size[0] - 1):
        for j in range(1, img.size[1] - 1):
            if pixels[i, j] == 255:
                continue
            pixel_cnt = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    if k != 0 and l != 0:
                        continue
                    pixel_cnt += (pixels[i + k, j + l] == 0)
            if pixel_cnt == 1:
                del_pos.append([i + k, j + l])
    for pos in del_pos:
        pixels[pos[0], pos[1]] = 255

    del_pos = []
    for i in range(1, img.size[0] - 1):
        for j in range(1, img.size[1] - 1):
            pixel_cnt = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    pixel_cnt += (pixels[i + k, j + l] == 0)
            if pixel_cnt <= threshold:
                del_pos.append([i + k, j + l])
    for pos in del_pos:
        pixels[pos[0], pos[1]] = 255
    return img

def letter_split(img):
    letters = []
    cur = -1
    pixels = img.load()
    for i in range(img.size[0]):
        pixel_cnt = 0
        for j in range(img.size[1]):
            pixel_cnt += (pixels[i, j] != 255)
        if pixel_cnt == 0 and cur != -1:
            if i - cur >= 14:
                letters.append([cur, int((cur + i) / 2)])
                letters.append([int((cur + i) / 2), i])
            else:
                letters.append([cur, i])
            cur = -1
        elif pixel_cnt > 0 and cur == -1:
            cur = i
    letters = sorted(letters, reverse=True, key=lambda k: k[1] - k[0])
    letters = sorted(letters[:4])
    return letters

def to_number(letter):
    if letter >= "0" and letter <= "9":
        return ord(letter) - ord("0")
    return ord(letter) - ord("A") + 10
