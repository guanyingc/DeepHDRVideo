"""
Util functions for image processing
Some functions are redundant in this repository
"""
import sys
import os
import cv2
import matplotlib
import numpy as np
from imageio import imread, imsave
from skimage.transform import resize
from matplotlib import cm

##### Misc #####
def raise_not_defined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)


def save_results_compact(name, results, width):
    intv = 5
    num = len(results) 
    w_n = int(width)
    h_n = int(np.ceil(float(num) / w_n))
    big_img = np.zeros(0)
    fix_h = 0; fix_w = 0
    for count, img in enumerate(results):
        if isinstance(img, (bool)) and img == False:
            print('Saving Flase, continue')
            continue
        if img.ndim < 3:
            img = np.tile(img.reshape(img.shape[0], img.shape[1], 1), (1,1,3))
        if img.dtype == np.uint8:
            img = img.astype(float)
        if img.max() <= 1:
            img = img * 255.0
        h, w, c = img.shape
        if count == 0:
           big_img = np.zeros((h_n*h + (h_n-1)*intv, w_n*w + (w_n-1)*intv, 3), dtype=np.uint8)
           fix_h = h;
           fix_w = w
        if h != fix_h or w != fix_w:
            #img = cv2.resize(img, (fix_h, fix_w))
            img = resize(img, (fix_h, fix_w), order=1, mode='reflect')
        h_idx = int(count / w_n);
        w_idx = count % w_n
        h_start = h_idx * (fix_h + intv)
        w_start = w_idx * (fix_w + intv)
        big_img[h_start:h_start+fix_h, w_start:w_start+fix_w, :] = img
    imsave(name, big_img)


def save_results_separate(prefix, results):
    for key in results:
        save_name = prefix + '_' + key
        value = results[key]
        if substrInList(key, ['mask', 'rho', 'diff']):
            save_name += '.png'
        elif substrInList(key, ['flow']):
            save_name += '.flo'
        else:
            save_name += '.jpg'
        if substrInList(key, ['flow']):
            write_flow_binary(value, save_name)
        else:
            imsave(save_name, value)


def check_empty(a_list):
    if len(a_list) == 0:
        raise Exception('Empty list')


def get_image_stat(img, fmt='%.4f'):
    max_v, mean_v, min_v = img.max(), img.mean(), img.min()
    template = 'max: %s, mean: %s, min: %s' % (fmt, fmt, fmt)
    stat = template % (max_v, mean_v, min_v)
    return stat


def print_image(img, name, fmt='%.4f'):
    stat = get_image_stat(img, fmt)
    print('%s, %s' % (name, stat))


###### Image Loading ######
def read_16bit_tif(img_name, crf=None):
    img = cv2.imread(img_name, -1) #/ 65535.0 # normalize to [0, 1]
    img = img[:, :, [2, 1, 0]] # BGR to RGB
    #print('before', img.max(), img.mean(), img.min())
    if crf is not None:
        img = reverse_crf(img, crf)
        img = img / crf.max()
    else:
        img = img / 65535.0
    #print('after', img.max(), img.mean(), img.min())
    return img


def read_hdr(filename, use_cv2=True):
    ext = os.path.splitext(filename)[1]
    if use_cv2:
        hdr = cv2.imread(filename, -1)[:,:,::-1].clip(0)
    elif ext == '.exr':
        hdr = read_exr(filename) 
    elif ext == '.hdr':
        hdr = cv2.imread(filename, -1)
    elif ext == '.npy':
        hdr = np.load(filenmae) 
    else:
        raise_not_defined()
    return hdr


def load_mask(mask_name):
    mask = imread(os.path.join(mask_name))
    if mask.ndim < 3:
        mask = mask.reshape(h, w, 1).repeat(3, 2)
    return mask


def read_img_list_from_dir(input_dir, exts=['.jpg'], sort=False, add_path=True):
    f_names = os.listdir(input_dir)
    img_names = []
    for ext in exts:
        img_names += [f for f in f_names if f.endswith(ext)]
    check_empty(img_names)
    if add_path: 
        img_list = ['%s' % os.path.join(input_dir, name) for name in img_names]
    else:
        img_list = img_names
    if sort == True:
        img_list.sort(key=natural_keys)
    return img_list


def read_img_from_list(img_list, verbose=True):
    if verbose:
        print('Reading images from list')
    check_empty(img_list)
    imgs = []
    for i, img_name in enumerate(img_list):
        I = imread(img_name)
        imgs.append(I)
    return imgs


###### Image editing ######
def rgb_to_gray(rgb):
    if len(rgb.shape) <= 2:
        return rgb
    return np.uint8(np.dot(rgb[...,:], [0.299, 0.587, 0.114]))


def rgb_to_gray3C(rgb):
    if len(rgb.shape) <= 2:
        return rgb
    gray_img = np.uint8(np.dot(rgb[...,:], [0.299, 0.587, 0.114]))
    h, w = gray_img.shape
    return np.tile(gray_img.reshape(h, w, 1), (1, 1, 3))


def list_rgb_to_gray(imgs):
    print('Converting list of rgb images to gray images')
    for i in range(len(imgs)):
        imgs[i] = rgb_to_gray(imgs[i])


def crop_img_border(img, border=10):
    h, w, c = img.shape
    img = img[border: h - border, border: w - border, :]
    return img


def reverse_crf(img, crf):
    img = img.astype(int)
    out = img.astype(float)
    for i in range(img.shape[2]):
        out[:,:,i] = crf[:,i][img[:,:,i]]
    return out


def colorNormalize(img): 
    h, w, c = img.shape
    img = img.reshape(-1, 3)
    mean = img.mean(0); 
    std = img.std(0) + 1e-10
    img = (img - mean) / std
    img = img.reshape(h, w, c)
    return img


##### Image Saving #####
def save_uint8(name, img):
    if img.dtype != np.uint8:
        img = (img.clip(0, 1) * 255).astype(np.uint8)
    imsave(name, img)


def save_uint16(img_name, img):
    """img in [0, 1]"""
    img = img.clip(0, 1) * 65535
    img = img[:,:,[2,1,0]].astype(np.uint16)
    cv2.imwrite(img_name, img)


##### HDR related #####
def ldr_to_hdr(img, expo, gamma=2.2):
    img = img.clip(0, 1)
    img = np.power(img, gamma) # linearize
    img /= expo
    return img


def hdr_to_ldr(img, expo, gamma=2.2):
    img = np.power(img * expo, 1.0 / gamma)
    img = img.clip(0, 1)
    return img


def ldr_to_ldr(img, expo_l2h, expo_h2l):
    if expo_l2h == expo_h2l:
        return img
    img = ldr_to_hdr(img, expo_l2h)
    img = hdr_to_ldr(img, expo_h2l)
    return img

def ldr_to_ldr_v2(img, expo_l2h, expo_h2l):
    img = img.clip(0, 1)
    if expo_l2h == expo_h2l:
        return img
    gain = np.power(expo_h2l / expo_l2h, 1.0/2.2)
    #print(expo_h2l, expo_l2h, gain)
    img = (img * gain).clip(0, 1)
    #print('gain', gain)
    return img


def save_hdr(name, hdr):
    print(name)
    hdr = hdr[:, :, [2, 1, 0]].astype(np.float32)
    cv2.imwrite(name, hdr)


def mulog_transform(in_tensor, mu=5000.0):
    denom = np.log(1.0 + mu)
    out_tensor = np.log(1.0 + mu * in_tensor) / denom 
    return out_tensor


def weightFunc(img):
    img = img.clip(0, 1)
    w = np.zeros(img.shape)
    mask1 = img >= 0.5
    w[mask1] = 1 - img[mask1]
    
    mask2 = img < 0.5
    w[mask2] = img[mask2]
    w /= 0.5
    return w


def merge_hdr(ldr_imgs, expos):
    sum_img = np.zeros(ldr_imgs[0].shape, dtype=np.float32)
    sum_w = np.zeros(ldr_imgs[0].shape)
    for i in range(len(ldr_imgs)):
        w = weightFunc(ldr_imgs[i])
        print_image(w, 'w')
        linear_img = ldr_to_hdr(ldr_imgs[i], expos[i])
        print_image(linear_img, 'linear')
        sum_w += w
        sum_img += linear_img * w
    print_image(sum_w, 'sum_w')
    print_image(sum_img, 'sum_img')
    out_img = sum_img / (sum_w + 1e-8)
    print_image(out_img, 'out_img')
    out_img = sum_img / (sum_w + 1e-8)
    saturation = sum_w < 1e-4
    return out_img


def tonemap(img):
    tonemap = cv2.createTonemapReinhard(2.2, 0, 1, 0) # gamam, intensity, light_adapt, color_adapt
    tm_img =tonemap.process(img)
    return tm_img


def tonemap_reinhard(hdr_name, gamma=0.85, key_f=0.10, sat_f=1.60, log_sum_prev=None):
    epsilon = 1e-6
    hdr = read_hdr(hdr_name)[:,:,::-1]
    img_lab = color.rgb2lab(hdr).astype(np.float32)
    h, w, _ = img_lab.shape
    if log_sum_prev is None:
        log_sum_prev = np.log(epsilon + img_lab[:,:,0].reshape(-1)).sum()
        key = key_f
    else:
        log_sum_cur = np.log(img_lab[:,:,0].reshape(-1) + epsilon).sum()
        log_avg_cur = np.exp(log_sum_cur / (h * w))
        log_avg_temp = np.exp((log_sum_cur + log_sum_prev) / (2.0 * h * w))

        key = key_f * log_avg_cur / log_avg_temp
        log_sum_prev = log_sum_cur

    save_path = hdr_name[:-4] + '_TM.tif'
    #save_path = os.path.join(out_dir, save_hdr_name)
    command = 'luminance-hdr-cli -l %s -t reinhard02 -p key=%f -g %f -o %s' % (
           hdr_name, key, gamma, save_path)
    os.system(command)
    img = read_16bit_tif(save_path)
    os.remove(save_path)
    img_hsv = color.rgb2hsv(img).astype(np.float32)

    img_hsv[:, :, 1] = img_hsv[:, :, 1] * sat_f
    img = color.hsv2rgb(img_hsv) # slow
    return img, log_sum_prev


##### EXR related #####
def read_exr(filename):
    hdr_file = OpenEXR.InputFile(filename)
    img = to_array(hdr_file)
    return img


def exr_to_array(I, cname='RGB', mute=True):
    hw = I.header()['dataWindow']
    w = hw.max.x - hw.min.x + 1
    h = hw.max.y - hw.min.y + 1
    #import pdb; pdb.set_trace() 
    prefix = ''
    R = np.fromstring(I.channel(prefix + 'R'), np.float16).reshape(h, w)
    G = np.fromstring(I.channel(prefix + 'G'), np.float16).reshape(h, w)
    B = np.fromstring(I.channel(prefix + 'B'), np.float16).reshape(h, w)

    img = np.stack([R, G, B], 2).astype(np.float32)
    #import pdb; pdb.set_trace() 
    if not mute:
        print('[Extracting %s], max value: %f, min: %f, mean: %f' % 
                (cname, img.max(), img.min(), img.mean()))
    return img


def save_exr(save_name, I): # width, height
    h, w, c = I.shape
    I = I.astype(np.float16).reshape(-1, c)
    r = I[:,0].tostring()
    g = I[:,1].tostring()
    b = I[:,2].tostring()
    header = OpenEXR.Header(w, h)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    header['channels'] = dict([(c, half_chan) for c in ['R', 'G', 'B']])
    exr = OpenEXR.OutputFile(save_name, header)
    exr.writePixels({'R': r, 'G': g, 'B': b})


# Optical flow related
def flow_to_map(F_mag, F_dir):
    sz = F_mag.shape
    flow_color = np.zeros((sz[0], sz[1], 3), dtype=float)
    flow_color[:,:,0] = (F_dir + np.pi) / (2 * np.pi)
    #f_dir = (F_dir + np.pi) / (2 * np.pi)
    #flow_color[:,:,1] = (F_mag / (F_mag.shape[1] * 0.5)).clip(0, 1)
    flow_color[:,:,1] = F_mag / F_mag.max()
    flow_color[:,:,2] = 1
    flow_color = matplotlib.colors.hsv_to_rgb(flow_color) #* 255
    return flow_color

def flow_to_color(flow):
    F_dx = flow[:,:,1].copy().astype(float)
    F_dy = flow[:,:,0].copy().astype(float)
    F_mag = np.sqrt(np.power(F_dx, 2) + np.power(F_dy, 2))
    F_dir = np.arctan2(F_dy, F_dx)
    flow_color = flow_to_map(F_mag, F_dir)
    return flow_color #.astype(np.uint8)

def read_flo_file(filename, short=True): # short indicates 16bit
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise Exception('Magic number incorrect: %s' % filename)
        h = int(np.fromfile(f, np.int32, count=1))
        w = int(np.fromfile(f, np.int32, count=1))
        
        if short:
            flow = np.fromfile(f, np.int16, count=h*w*2)
            #flow = flow.astype(np.float32)
        else:
            flow = np.fromfile(f, np.float32, count=h*w*2)
        flow = flow.reshape((h, w, 2)).astype(np.float32)
    return flow

def write_flow_binary(flow, filename, short=True):
    if short:
        flow = flow.astype(np.int16)
    else:
        flow = flow.astype(np.float32)
    with open(filename, 'wb') as f:
        magic = np.array([202021.25], dtype=np.float32) 
        h_w   = np.array([flow.shape[0], flow.shape[1]], dtype=np.int32)
        magic.tofile(f)
        h_w.tofile(f)
        flow.tofile(f)

def get_grid(h, w):
    x_grid = np.tile(np.linspace(0, w-1, w), (h, 1)).astype(float)
    y_grid = np.tile(np.linspace(0, h-1, h), (w, 1)).T.astype(float)
    return x_grid, y_grid

def warp_image(ref, flow, grid_x, grid_y):
    flow_x = np.clip(flow[:,:,1] + grid_x, 0, args.w-1)
    flow_y = np.clip(flow[:,:,0] + grid_y, 0, args.h-1)
    #print(flow_x.min(), flow_x.max(), flow_y.min(), flow_y.max())
    flow_x, flow_y = cv2.convertMaps(flow_x.astype(np.float32), flow_y.astype(np.float32), cv2.CV_32FC2) 
    warped_img = cv2.remap(ref, flow_x, flow_y, cv2.INTER_LINEAR)
    return warped_img

def colormap(diff, thres=0.1):
    diff = diff.clip(0, thres) / thres
    diff_map = cm.jet(diff)[:, :, :3]
    #diff = (diff * 255).astype(np.uint8)
    #diff_map = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    #diff_map = cv2.cvtColor(diff_map, cv2.COLOR_BGR2RGB)
    return diff_map.copy()

def apply_gamma(image, gamma=2.2):
    image = image.clip(1e-8, 1)
    image = np.power(image, 1.0 / gamma)
    return image
