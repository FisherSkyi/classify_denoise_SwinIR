import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
from PIL import Image
from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='color_dn', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car, color_jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='./model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth ',)
    # parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)

    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    psnr, ssim, psnr_y, ssim_y = 0, 0, 0, 0

    # restrict to image files only
    image_exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.ppm']
    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob.glob(os.path.join(folder, ext)))

    for idx, path in enumerate(sorted(image_paths)):
        # read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        # img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = np.transpose(img_lq, (2, 0, 1))  # HWC-RGB → CHW-RGB (no channel swapping)
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
            img_gt = np.squeeze(img_gt)

            psnr = util.calculate_psnr(output, img_gt, crop_border=border)
            ssim = util.calculate_ssim(output, img_gt, crop_border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:  # RGB image
                psnr_y = util.calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
                ssim_y = util.calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
            print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; '
                  'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}.'.
                  format(idx, imgname, psnr, ssim, psnr_y, ssim_y))
        else:
            print('Testing {:d} {:20s}'.format(idx, imgname))

    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(save_dir, ave_psnr, ave_ssim))
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))

    csv_path = os.path.join(save_dir, "results.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'PSNR', 'SSIM', 'PSNR_Y', 'SSIM_Y'])

        for i in range(len(test_results['psnr'])):
            writer.writerow([
                os.path.basename(image_paths[i]),
                f"{test_results['psnr'][i]:.4f}",
                f"{test_results['ssim'][i]:.4f}",
                f"{test_results['psnr_y'][i]:.4f}" if i < len(test_results['psnr_y']) else '',
                f"{test_results['ssim_y'][i]:.4f}" if i < len(test_results['ssim_y']) else ''
            ])
    print(f"Results saved to {csv_path}")


def define_model(args):
    if args.task == 'color_dn':
        model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    return model


def setup(args):
    if args.task in ['color_dn']:
        # class_name = os.path.basename(os.path.normpath(args.folder_gt))  # e.g., "00000"
        save_dir = f'results/test_denoise_noise{args.noise}'
        folder = args.folder_gt
        border = 0
        window_size = 8
    return folder, save_dir, border, window_size

def add_gaussian_noise(img: Image.Image, std: float) -> Image.Image:
    """
    Return a copy of *img* with additive Gaussian noise.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0.0, std, rgb.shape)
    noisy = np.clip(rgb + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")

def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    if args.task in ['color_dn']:
        # img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        # np.random.seed(seed=0)
        # img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)
        img_gt_pil = Image.open(path).convert("RGB")
        img_lq_pil = add_gaussian_noise(img_gt_pil, std=args.noise)
        img_gt = np.asarray(img_gt_pil, dtype=np.float32) / 255.0
        img_lq = np.asarray(img_lq_pil, dtype=np.float32) / 255.0
    return imgname, img_lq, img_gt


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()
