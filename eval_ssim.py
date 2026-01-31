import os
import torch
import numpy as np
import json
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ssim

def calculate_ssim_for_nested_folders(main_dir):
    results = {}
    all_ssim = []

    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    os.makedirs(main_dir, exist_ok=True)
    output_path = os.path.join(main_dir, "ssim_results.json")


    subdirs = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])

    
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)

        original_dir = os.path.join(subdir_path, "original")
        adv_dir = os.path.join(subdir_path, "adv")

        
        original_files = sorted(
            [f for f in os.listdir(original_dir) if f.lower().endswith(('.jpg', '.png'))],
            key=lambda x: int(''.join(filter(str.isdigit, x)))
        )
        adv_files = sorted(
            [f for f in os.listdir(adv_dir) if f.lower().endswith(('.jpg', '.png'))],
            key=lambda x: int(''.join(filter(str.isdigit, x)))
        )

    

        subdir_results = {}
        ssim_values = []

        for filename in original_files:
            if subdir in results and filename in results[subdir]:
                ssim_val = results[subdir][filename]
                subdir_results[filename] = ssim_val
                ssim_values.append(ssim_val)
                all_ssim.append(ssim_val)
                print(f"Have Calculated: {subdir}/{filename} - SSIM={ssim_val:.4f}")
                continue

            try:
                original_img = transform(
                    Image.open(os.path.join(original_dir, filename))
                ).unsqueeze(0)  # [1,C,H,W]
                adv_img = transform(
                    Image.open(os.path.join(adv_dir, filename))
                ).unsqueeze(0)

                
                current_ssim = ssim(original_img, adv_img, data_range=1.0, size_average=False).item()
                subdir_results[filename] = current_ssim
                ssim_values.append(current_ssim)
                all_ssim.append(current_ssim)
                
            except Exception as e:
                print(f"{subdir}/{filename} error: {str(e)}")
                continue

        if ssim_values:
            subdir_mean = np.mean(ssim_values)
            subdir_results["mean"] = subdir_mean
            results[subdir] = subdir_results


            results["global_mean"] = np.mean(all_ssim) if all_ssim else 0

            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
            # print(f"{subdir}:{results[subdir]['mean']}")
    for subdir, data in results.items():
        if subdir == "global_mean":
            print(f"\nGloabl_SSIM: {data:.4f}")
    print(f"\nSave in: {output_path}")
    return results

if __name__ == "__main__":
    main_directory = ""
    ssim_results = calculate_ssim_for_nested_folders(main_directory)

    for subdir, data in ssim_results.items():
        if subdir == "global_mean":
            print(f"\nGlobal_SSIM: {data:.4f}")
        else:
            print(f"\n{subdir} results:")
            for filename, score in data.items():
                if filename != "mean":
                    print(f"  {filename}: {score:.4f}")
            print(f" average: {data['mean']:.4f}")

