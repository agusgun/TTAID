import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_batch_psnr(gt_tensor, output_tensor, mode='avg'):
    # both parameters are in the form of tensor of size: BS, C, H, W
    
    if mode == 'avg':
        gt_np = gt_tensor.cpu().numpy().astype(np.float32)
        output_np = output_tensor.cpu().numpy().astype(np.float32)

        bs = gt_np.shape[0]
        psnr = 0
        for i in range(bs):
            gt_im = gt_np[i, :, :, :]
            output_im = output_np[i, :, :, :]
            
            gt_im = gt_im.transpose((1, 2, 0))
            output_im = output_im.transpose((1, 2, 0))

            psnr += peak_signal_noise_ratio(gt_im, output_im, data_range=1.)
        return psnr / bs, bs
    else:
        raise NotImplementedError

def calculate_batch_ssim(gt_tensor, output_tensor, mode='avg'):
    if mode == 'avg':
        gt_np = gt_tensor.cpu().numpy().astype(np.float32)
        output_np = output_tensor.cpu().numpy().astype(np.float32)

        bs = gt_np.shape[0]
        ssim = 0
        for i in range(bs):
            gt_im = gt_np[i, :, :, :]
            output_im = output_np[i, :, :, :]
            gt_im = gt_im.transpose((1, 2, 0))
            output_im = output_im.transpose((1, 2, 0))
            
            ssim += structural_similarity(gt_im, output_im, data_range=1., multichannel=True)
        
        return ssim / bs, bs
    else:
        raise NotImplementedError