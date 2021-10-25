import imageio
import os
import torchvision.utils as v_utils
import torch


def plot_samples_per_epoch(gen_batch, output_dir, epoch):
    """
    Plot and save output samples per epoch
    """
    fname = "samples_epoch_{:d}.png".format(epoch)
    fpath = os.path.join(output_dir, fname)

    v_utils.save_image(gen_batch, fpath, nrow=4, padding=2, normalize=True, value_range=(0, 1))
    return imageio.imread(fpath)

def plot_val_samples(gen_batch, output_dir, fname):
    """
    Plot and dsave output samples for validations
    """
    fpath = os.path.join(output_dir, fname)

    v_utils.save_image(gen_batch, fpath, nrow=4, padding=2, normalize=True, value_range=(0, 1))
    return imageio.imread(fpath)

def plot_image(img, output_dir, fname):
    """
    img in tensor format
    """

    fpath = os.path.join(output_dir, fname)

    v_utils.save_image(img, fpath, normalize=True, value_range=(0, 1))
    return imageio.imread(fpath)

def stitch(net, input_tensor, n_patches=[4, 4], output_channels=3, mask=None, output_index=None):
    """
    input tensor must have size 1, C, H, W
    net: network

    return stitched and saved image
    """

    # Stitching part
    extra_region_size = 32

    _, c, h, w = input_tensor.size() # for now bs must be one
    patch_size = [h // n_patches[0], w // n_patches[1]]

    output = torch.zeros((1, output_channels, h, w), dtype=input_tensor.dtype, device=input_tensor.device)
    for patch_idx_row in range(n_patches[0]):
        for patch_idx_col in range(n_patches[1]):
            top_left_position = (patch_idx_row * patch_size[0], patch_idx_col * patch_size[1])
            is_corner = (patch_idx_row == n_patches[0] - 1, patch_idx_col == n_patches[1] - 1)
            patch_mapping_position = _get_patch_mapping_position(
                input_tensor, patch_size, extra_region_size, top_left_position, is_corner)
            
            image_position = patch_mapping_position['image']
            image_patch_position = patch_mapping_position['image_patch_coordinate']
            patch_position = patch_mapping_position['internal_patch_coordinate']

            current_input = input_tensor[
                :, :, image_patch_position[0]:image_patch_position[2], 
                image_patch_position[1]:image_patch_position[3]]
            
            if mask is not None:
                current_mask = mask[
                    :, :, image_patch_position[0]:image_patch_position[2], 
                    image_patch_position[1]:image_patch_position[3]]
                current_output = net(current_input, current_mask)
            else:
                current_output = net(current_input)
            if output_index is not None:
                current_output = current_output[output_index]
            output[:, :, image_position[0]:image_position[2], image_position[1]:image_position[3]] = current_output[
                :, :, patch_position[0]:patch_position[2], patch_position[1]:patch_position[3]
            ]

    return output

def _get_patch_mapping_position(img_tensor, patch_size, extra_region_size, top_left_position, is_corner):
    """
    img_tensor: tensor - input image tensor
    patch_size: [int, int] - height and width (size) of each patch
    extra_region_size: int - the size of extra region for all of the positions (top, left, bottom, right)
    top_left_position: (int, int) - top and left position 
    is_corner: (bool, bool) - last patch based on (row, column) of the image
    """
    patch_size = patch_size.copy()
    _, _, img_h, img_w = img_tensor.size()

    if img_h < top_left_position[0] + patch_size[0]:
        patch_size[0] = img_h - top_left_position[0]
    if img_w < top_left_position[1] + patch_size[1]:
        patch_size[1] = img_w - top_left_position[1]
    
    patch_top_left_position = (abs(max(top_left_position[0] - extra_region_size, 0) - top_left_position[0]), 
        abs(max(top_left_position[1] - extra_region_size, 0) - top_left_position[1]))
    patch_bottom_right_position = (patch_top_left_position[0] + patch_size[0], 
        patch_top_left_position[1] + patch_size[1])

    mapping_position = {
        'image': [ # tlbr
            top_left_position[0],
            top_left_position[1],
            top_left_position[0] + patch_size[0], # patch size already resized 
            top_left_position[1] + patch_size[1] # patch size already resized
        ],
        'internal_patch_coordinate': [ # tlbr
            patch_top_left_position[0],
            patch_top_left_position[1],
            patch_bottom_right_position[0],
            patch_bottom_right_position[1]
        ],
        'image_patch_coordinate': [
            max(top_left_position[0] - extra_region_size, 0),
            max(top_left_position[1] - extra_region_size, 0),
            min(top_left_position[0] + patch_size[0] + extra_region_size, img_h),
            min(top_left_position[1] + patch_size[1] + extra_region_size, img_w)
        ],
        'patch_size': [
            patch_size[0],
            patch_size[1]
        ]
    }

    if is_corner[0]: # last patch (row-wise)
        mapping_position['internal_patch_coordinate'][2] += img_h - (top_left_position[0] + patch_size[0]) 
        mapping_position['image'][2] = img_h
        

    if is_corner[1]: # last patch (column-wise)
        mapping_position['internal_patch_coordinate'][3] += img_w - (top_left_position[1] + patch_size[1])
        mapping_position['image'][3] = img_w
    
    return mapping_position
        
def make_gif(output_dir, epochs):
    """
    Make a gif from multiple images of epochs
    """
    gen_image_plots = []
    for epoch in range(epochs + 1):
        fname = "samples_epoch_{:d}.png".format(epoch)
        img_epoch = imageio.imread(os.path.join(output_dir, fname))
        gen_image_plots.append(img_epoch)

    imageio.mimsave(os.path.join(output_dir, 'animation_epochs_{:d}.gif'.format(epochs)), gen_image_plots, fps=2)

if __name__ == "__main__":
    net = torch.nn.Identity()
    input_tensor = torch.ones((1, 3, 513, 513))
    output_dir = 'data'
    fname = 'fname.png'

    print(_get_patch_mapping_position(input_tensor, [128, 128], 32, (0, 440), (False, True)))
    print(_get_patch_mapping_position(input_tensor, [128, 128], 32, (384, 384), (False, False)))

    out = stitch(net, input_tensor, output_dir, fname)