"""
Compute similarity transformation matrices between neighbouring frames for a video

example usage:
python utils/compute_nb_transformation_video.py --in_dir /path/to/your/dataset/ --scene_list /your/scene_list --crf gamma
"""
import argparse
import glob
import os
import shutil
import cv2
import numpy as np
import scipy.io as sio
from imageio import imsave


class VideoGlobalAligner(object):
    def __init__(self, args):
        self.args = args
    
    def process_scenes(self):
        args = self.args
        scenes = self.load_scenes_list(args)
        
        if args.crf == 'gamma':
            self.crf = None
        else:
            self.crf = sio.loadmat(args.crf)['BaslerCRF']

        for i_s, scene in enumerate(scenes):
            print('[%d/%d]: %s' % (i_s, len(scenes), scene))

            scene_dir = os.path.join(args.in_dir, scene)
            img_names, expos = self.load_scene_data(scene_dir)
            
            self.process_scene(scene_dir, img_names, expos)

    def read_list(self, list_path,ignore_head=False, sort=False):
        lists = []
        with open(list_path) as f:
            lists = f.read().splitlines()
        if ignore_head:
            lists = lists[1:]
        if sort:
            lists.sort(key=natural_keys)
        return lists

    def load_scenes_list(self, args):
        scene_list = os.path.join(args.in_dir, args.scene_list)
        if os.path.exists(scene_list):
            print('Loading Scene list: %s' % scene_list)
            scenes = self.read_list(scene_list)
        else:
            print('Glob scene list: %s' % scene_list)
            files = sorted(glob.glob(os.path.join(in_dir, '*')))
            scenes = [os.path.basename(file_name) for file_name in files if os.path.isdir(file_name)] # filter no directory file
        #scenes = np.genfromtxt(os.path.join(args.in_dir, args.scene_list), dtype='str')
        return scenes

    def load_scene_data(self, scene_dir, img_list='img_list.txt'):
        img_list_name = os.path.join(scene_dir, img_list)
        img_names = np.genfromtxt(os.path.join(scene_dir, img_list), dtype='str')
        img_names = [os.path.join(scene_dir, img_name) for img_name in img_names]
        expos = np.genfromtxt(os.path.join(scene_dir, 'Exposures.txt'), dtype=float)
        expos = np.power(2.0, expos - expos.min()) # from stop to second
        return img_names, expos

    def process_scene(self, scene_dir, img_names, expos):
        nexps = len(expos)
        if nexps == 2:
            nbrs = 1  # number of neighbours that needs to compute affine transformation matrix
        elif nexps == 3:
            nbrs = 2
        else:
            raise Exception('Unknown number of exposures: %d' % nexps)

        save_dir = os.path.join(scene_dir, 'Affine_Trans_Matrices')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(len(img_names)):
            if self.args.max_num > 0 and i >= self.args.max_num:
                break
            frame_name = os.path.basename(img_names[i])[:-4]

            if i % self.args.display == 0:
                print('\t [%d/%d] frames: %s' % (i, len(img_names), frame_name))

            cur = self.read_image(img_names[i], crf=self.crf)

            matches = []
            for j in range(i-nbrs, i):
                if j < 0:
                    matches.append(np.array([[1, 0, 0], [0, 1, 0]]).reshape(-1))
                    continue

                prev = self.read_image(img_names[j], crf=self.crf)

                transform, match = self.compute_matches(cur, prev, expos[i % nexps], expos[j % nexps])
                matches.append(transform.reshape(-1))
                imsave(os.path.join(save_dir, frame_name + '_%03d_%01d.jpg' % (i, j)), match)

            for j in range(i+1, i+nbrs+1):
                if j >= len(img_names):
                    matches.append(np.array([[1, 0, 0], [0, 1, 0]]).reshape(-1))
                    continue

                nxt = self.read_image(img_names[j], crf=self.crf)
                transform, match = self.compute_matches(cur, nxt, expos[i%nexps], expos[j%nexps])
                matches.append(transform.reshape(-1))
    
            matches = np.stack(matches, 0).astype(np.float32)
            print(matches.shape, matches)
            np.savetxt(os.path.join(save_dir, frame_name + '_match.txt'), matches)
    
    def reverse_crf(self, img, crf):
        img = img.astype(int)
        out = img.astype(float)
        for i in range(img.shape[2]):
            out[:,:,i] = crf[:,i][img[:,:,i]]
        return out

    def read_16bit_tif(self, img_name, crf=None):
        img = cv2.imread(img_name, -1) #/ 65535.0 # normalize to [0, 1]
        img = img[:, :, [2, 1, 0]] # BGR to RGB
        if crf is not None:
            img = self.reverse_crf(img, crf)
            img = img / crf.max()
        else:
            img = img / 65535.0
        return img
    
    def apply_gamma(self, image, gamma=2.2):
        image = image.clip(1e-8, 1)
        image = np.power(image, 1.0 / gamma)
        return image

    def read_image(self, img_name, crf=None):
        if crf is None:
            #print('Gamma CRF')
            ext = img_name[-4:]
            if ext in ['.jpg', '.png']:
                img = cv2.imread(img_name, -1).astype(np.float32) / 255.0
            elif ext in ['.tif']:
                img = cv2.imread(img_name, -1).astype(np.float32) / 65536.0
            else:
                raise Exception('Unknown file extension: %s' % ext)
        else:
            linear_img = self.read_16bit_tif(img_name, crf)
            img = self.apply_gamma(linear_img)

        return img

    def ldr_to_ldr(self, img, expo_l2h, expo_h2l):
        img = img.clip(0, 1)
        if expo_l2h == expo_h2l:
            return img
        gain = np.power(expo_h2l / expo_l2h, 1.0/2.2)
        img = (img * gain).clip(0, 1)
        return img

    def adjust_imgs_expos(self, imgs, expos):
        max_expos = max(expos)
        adjust_imgs = []
        for i in range(len(imgs)):
            adjust_img = self.ldr_to_ldr(imgs[i], expos[i], max_expos)
            adjust_imgs.append(adjust_img)
        return adjust_imgs

    def compute_transform(self, img1, img2, draw_match=True):
        height, width, channels = img2.shape
        keypoints, descriptors = self.get_descriptor([img1, img2])
        transform_prev, _, matches = self.compute_affine_partial2d(
                keypoints[0], descriptors[0], keypoints[1], descriptors[1])
        if draw_match:
            # Draw top matches
            im1_u8, im2_u8 = (img1 * 255).astype(np.uint8), (img2 * 255).astype(np.uint8)
            imMatches = cv2.drawMatches(im1_u8, keypoints[0], im2_u8, keypoints[1], matches, None)
        else:
            imgMatches = None
        return transform_prev, imMatches

    def get_descriptor(self, imgs):
        imgs_gray = []
        keypoints = []
        descriptors = []
        MAX_MATCHES = 500
        orb = cv2.ORB_create(MAX_MATCHES)
        for img in imgs:
            img = (img * 255.0).astype(np.uint8)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            kps, dscps = orb.detectAndCompute(img_gray, None)
            keypoints.append(kps)
            descriptors.append(dscps)
        return keypoints, descriptors

    def compute_affine_partial2d(self, keypoints1, descriptors1, keypoints2, descriptors2):
        GOOD_MATCH_PERCENT = 0.15
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        matches.sort(key=lambda x: x.distance, reverse=False)
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        similarity, inliers = cv2.estimateAffinePartial2D(points1, points2, cv2.RANSAC)
        return similarity, inliers, matches

    def compute_matches(self, img1, img2, expo1, expo2):
        expo_adjs_imgs = self.adjust_imgs_expos([img1, img2], [expo1, expo2])
        transform, match = self.compute_transform(expo_adjs_imgs[0], expo_adjs_imgs[1])
        return transform, match


def main(args):
    processer = VideoGlobalAligner(args)
    processer.process_scenes()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default='.')
    parser.add_argument('--scene_list', default='scenes.txt') # 2Exp_scenes.txt, 3Exp_scenes.txt for TOG13
    parser.add_argument('--img_list', default='img_hdr_list.txt')
    parser.add_argument('--exposure', default='Exposures.txt')

    parser.add_argument('--crf', default='data/TOG13_Dynamic_Dataset/BaslerCRF.mat', help='use gamma if you do not know crf')
    parser.add_argument('--max_num', default=-1, type=int) # 5 for 3exposures
    parser.add_argument('--display', default=1, type=int) # 5 for 3exposures
    parser.add_argument('--gamma', default=2.2, type=float)
    parser.add_argument('--ext', default='png')
    args = parser.parse_args()
    return args 


if __name__ == '__main__':
    args = parse_args()
    main(args)
