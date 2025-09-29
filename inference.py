import os
from os.path import basename, splitext, join
import numpy as np
import SimpleITK as sitk
import torch
from glob import glob
from sam2.build_sam import build_sam2_video_predictor_npz
from PIL import Image, ImageDraw


input_folder = "/home/ricciardi.giuseppe@ihsr.dom/Desktop/code/medsam2/MedSAM2/data/test/original_cropped"
checkpoint_folder = "/home/ricciardi.giuseppe@ihsr.dom/Desktop/code/medsam2/MedSAM2/data/test/checkpoints"
model_cfg = "./configs/sam2.1_hiera_t512.yaml"
output_root = "/home/ricciardi.giuseppe@ihsr.dom/Desktop/code/medsam2/MedSAM2/data/test/inference"
mask_folder = "/home/ricciardi.giuseppe@ihsr.dom/Desktop/code/medsam2/MedSAM2/data/test/masks"


def save_example_bbox_overlay(img_path, bbox, frame_idx, save_dir, original_size, image_size=512):
    """
    Save an overlay image with bbox drawn on a resized slice.
    bbox is in original resolution, so scale it to image_size for visualization.
    """
    nii_image = sitk.ReadImage(img_path)
    nii_image_data = sitk.GetArrayFromImage(nii_image)
    
    lower_bound, upper_bound = -300, 250
    slice_img = nii_image_data[frame_idx]
    slice_img = np.clip(slice_img, lower_bound, upper_bound)
    slice_img = ((slice_img - lower_bound) / (upper_bound - lower_bound) * 255.0).astype(np.uint8)

    img_pil = Image.fromarray(slice_img)
    img_resized = img_pil.resize((image_size, image_size))
    
    scaled_bbox = scale_bbox_to_512(bbox, original_size, image_size)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #save_path = os.path.join(save_dir, f"slice_{frame_idx}_bbox.png")
    #save_slice_with_bbox(np.array(img_resized), scaled_bbox, save_path)
    #print(f"Saved bbox overlay image: {save_path}")


def scale_bbox_to_512(bbox, original_size, target_size=512):
    """
    Scale bbox coordinates from original image size to target size (usually 512).
    bbox = [x1, y1, x2, y2]
    original_size = (height, width)
    """
    x1, y1, x2, y2 = bbox
    h, w = original_size
    scale_x = target_size / w
    scale_y = target_size / h
    return [
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    ]


def save_slice_with_bbox(image_2d, bbox, save_path):
    img_pil = Image.fromarray(image_2d).convert("RGB")
    draw = ImageDraw.Draw(img_pil)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    img_pil.save(save_path)


def resize_mask_slice(mask_2d, new_size=(512, 512)):
    mask_img = Image.fromarray(mask_2d.astype(np.uint8))
    mask_resized = mask_img.resize(new_size, resample=Image.NEAREST)
    return np.array(mask_resized)


def extract_bbox_from_mask(mask_2d):
    y_indices, x_indices = np.where(mask_2d > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    x1, y1 = np.min(x_indices), np.min(y_indices)
    x2, y2 = np.max(x_indices), np.max(y_indices)
    return [x1, y1, x2, y2]


def get_best_slice_and_bbox_original_res(mask_array):
    """
    Select the slice with the largest foreground object and return bbox in original resolution.
    """
    max_area = 0
    best_idx = -1
    best_bbox = None
    last_idx = -1
    z_found = False
    for i in range(mask_array.shape[0]):
        mask_2d = mask_array[i]
        bbox = extract_bbox_from_mask(mask_2d) 
        if bbox is not None:
            last_idx = i
            if z_found != True:
                best_idx = i
                z_found = True
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_bbox = bbox

    if best_bbox is None:
        raise ValueError("No object found in the mask volume.")

    return best_idx, best_bbox, last_idx


def load_nii_as_preprocessed_rgb_tensor(file_path, image_size=512):
    """
    Load nii image, clip and normalize, resize slices to 512x512 and convert to normalized RGB tensor.
    """
    nii_image = sitk.ReadImage(file_path)
    nii_image_data = sitk.GetArrayFromImage(nii_image)

    lower_bound, upper_bound = -300, 250
    nii_image_data = np.clip(nii_image_data, lower_bound, upper_bound)
    nii_image_data = (nii_image_data - lower_bound) / (upper_bound - lower_bound) * 255.0
    nii_image_data = nii_image_data.astype(np.uint8)

    d, h, w = nii_image_data.shape
    print("ORIGINAL IMAGE SHAPE: ", (d, h, w))
    resized_array = np.zeros((d, 3, image_size, image_size))
    for i in range(d):
        img_pil = Image.fromarray(nii_image_data[i])
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)
        resized_array[i] = img_array

    resized_array = resized_array / 255.0
    tensor = torch.from_numpy(resized_array).float().cuda()

    img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None].cuda()
    tensor = (tensor - img_mean) / img_std

    return tensor, nii_image


def run_inference_and_save(predictor, img_path, patient_name, input_tensor, reference_nii, save_path, mask_path=None, video_height=None, video_width=None):
    """
    Run model inference using bbox from mask_path if available, otherwise use full bbox.
    Propagate forward and backward to get segmentation.
    """
    num_slices = input_tensor.shape[0]
    
    if video_height is None or video_width is None:
        size = reference_nii.GetSize()
        video_width, video_height = size[0], size[1]

    inference_state = predictor.init_state(input_tensor, video_height, video_width)

    if mask_path is not None and os.path.exists(mask_path):
        mask_nii = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask_nii)
        print("ORIGINAL MASK SHAPE: ", mask_array.shape)
        frame_idx, bbox, last_idx = get_best_slice_and_bbox_original_res(mask_array)
        box = torch.tensor(bbox, dtype=torch.int).cuda()
        print(f"Using slice {frame_idx} with bbox: {bbox}")
    else:
        frame_idx = num_slices // 2
        box = torch.tensor([0, 0, video_width, video_height], dtype=torch.int).cuda()
        print(f"Using default slice {frame_idx} with full image bbox.")

    original_size = (video_height, video_width)
    save_example_bbox_overlay(img_path, bbox, frame_idx, save_dir=f"./test/overlays/{patient_name}", original_size=original_size, image_size=512)

    _, _, _ = predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=frame_idx, obj_id=1, box=box)
    segs_up = np.zeros((num_slices, video_height, video_width), dtype=np.uint8)
    segs_down = np.zeros((num_slices, video_height, video_width), dtype=np.uint8)
    for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
        segs_up[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0].astype(np.uint8)

    predictor.reset_state(inference_state)
    _, _, _ = predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=last_idx, obj_id=1, box=box)
    for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
        segs_down[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0].astype(np.uint8)

    predictor.reset_state(inference_state)

    segs = np.logical_and(segs_up, segs_down).astype(np.uint8)
    print(f"Nonzero segmentation voxels: {np.sum(segs > 0)}")
    sitk_mask = sitk.GetImageFromArray(segs)
    sitk_mask.CopyInformation(reference_nii)
    sitk.WriteImage(sitk_mask, save_path)


def extract_patient_id(filename):
    name = basename(filename)
    return name


def extract_model_name(checkpoint_path):
    return splitext(basename(checkpoint_path))[0]


def main():
    print(model_cfg)
    input_files = sorted(glob(join(input_folder, "*.nii.gz")))
    checkpoints = sorted(glob(join(checkpoint_folder, "*.pt")))

    torch.set_float32_matmul_precision('high')
    torch.manual_seed(2024)
    torch.cuda.manual_seed(2024)
    np.random.seed(2024)

    for checkpoint in checkpoints:
        model_name = extract_model_name(checkpoint)
        print(f"\nUsing model: {model_name}")
        predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)

        for img_path in input_files:
            patient_id = extract_patient_id(img_path)
            print(f"Processing {basename(img_path)} with {model_name}")
            input_tensor, reference_nii = load_nii_as_preprocessed_rgb_tensor(img_path)

            mask_path = join(mask_folder, basename(img_path))
            if not os.path.exists(mask_path):
                print(f"Mask not found for {img_path}, skipping.")
                continue

            save_dir = join(output_root, patient_id, model_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = join(save_dir, "segmentation.nii.gz")

            nii_image_data = sitk.GetArrayFromImage(reference_nii)
            _, h, w = nii_image_data.shape
            print(h,w)
            run_inference_and_save(
                predictor,
                img_path,
                patient_id,
                input_tensor,
                reference_nii,
                save_path,
                mask_path=mask_path,
                video_height=h,
                video_width=w
            )

if __name__ == "__main__":
    main()
