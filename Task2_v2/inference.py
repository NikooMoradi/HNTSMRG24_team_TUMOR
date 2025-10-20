from pathlib import Path
from glob import glob
import SimpleITK as sitk
import numpy as np
import subprocess
import os
import time

INPUT_PATH = Path("/input")  # For Docker
OUTPUT_PATH = Path("/output")  #

#INPUT_PATH = Path("test/input")  # If you want to run out of VScode local python uncomment, otherwise comment for Docker
#OUTPUT_PATH = Path("test/output")   

MODEL_INPUT_PATH = OUTPUT_PATH / "model_input"  # Now relative

NNUNET_OUTPUT_PATH = OUTPUT_PATH / "nnunet_output"  # Now relative
MEDNEXT_OUTPUT_PATH = OUTPUT_PATH / "mednext_output"  # Now relative

NNUNET_ENSEMBLE_OUTPUT_PATH = Path("nnunet_ensemble_output")
#NNUNET_ENSEMBLE_OUTPUT_PATH = OUTPUT_PATH / "nnunet_ensemble_output"  # working
MEDNEXT_ENSEMBLE_OUTPUT_PATH = OUTPUT_PATH / "mednext_ensemble_output"  # Now relative

#os.makedirs(NNUNET_ENSEMBLE_OUTPUT_PATH, exist_ok=True)

def run():
    start_time = time.time()
    print("Start time:", time.ctime(start_time))

    """
    Main function to read input, process the data, and write output.
    """
    os.environ['nnUNet_results'] = 'models/nnUNet_results/'  # Now relative
    os.environ['nnUNet_preprocessed'] = 'None'
    os.environ['nnUNet_raw'] = 'None'
    _show_torch_cuda_info()

    input_dirs = [p for p in INPUT_PATH.iterdir() if p.is_dir()]
    for input_dir in input_dirs:
        # Convert and rename inputs (still the same)
        start_convert_inputs = time.time()
        convert_and_rename_inputs(input_dir)
        print(f"Time to convert and rename inputs: {time.time() - start_convert_inputs} seconds")

        # Full resolution inference (all folds handled at once)
        start_nn_fullres = time.time()
        nnunet_fullres_inference()  # Now handles all folds
        print(f"Time for nnUNet full resolution inference: {time.time() - start_nn_fullres} seconds")

        # Low resolution inference (all folds handled at once)
        start_nn_lowres = time.time()
        nnunet_lowres_inference()  # Now handles all folds
        print(f"Time for nnUNet low resolution inference: {time.time() - start_nn_lowres} seconds")

        # Cascade inference (all folds handled at once)
        start_nn_cascade = time.time()
        nnunet_cascade_inference()  # Now handles all folds
        print(f"Time for nnUNet cascade inference: {time.time() - start_nn_cascade} seconds")

        # Ensemble step (remains the same)
        start_nn_ensemble = time.time()
        nnunet_ensemble()
        print(f"Time for nnUNet ensemble: {time.time() - start_nn_ensemble} seconds")

        # Convert and save the final output (remains the same)
        start_convert_output = time.time()
        convert_and_save_final_output(input_dir.name)
        print(f"Time to convert and save final output: {time.time() - start_convert_output} seconds")

    total_time = time.time() - start_time
    print("Total execution time:", total_time, "seconds")
    return 0


def convert_and_rename_inputs(input_dir):
    #MODEL_INPUT_PATH.mkdir(parents=True, exist_ok=True)
    os.makedirs(MODEL_INPUT_PATH)
    name_mapping = {
        "registered-pre-rt-head-neck-segmentation": "concatenated_500_0001.nii.gz",
        "registered-pre-rt-head-neck": "concatenated_500_0000.nii.gz",
        "mid-rt-t2w-head-neck": "concatenated_500_0002.nii.gz",
    }

    for dir_name, output_filename in name_mapping.items():
        input_folder = f"{input_dir}/{dir_name}"
        input_file = glob(f"{input_folder}/*.mha")[0]  # Get the .mha file inside the folder
        output_file = f"{MODEL_INPUT_PATH}/{output_filename}"
        convert_mha_to_niigz(input_file, output_file)


def convert_mha_to_niigz(input_file, output_file):
    start_convert_mha = time.time()
    image = sitk.ReadImage(str(input_file))
    sitk.WriteImage(image, str(output_file))
    print(f"Time to convert MHA to NII.GZ: {time.time() - start_convert_mha} seconds")


def mednext_S3_fullres_inference(fold):
    start_mednext = time.time()
    output_folder = f"{MEDNEXT_OUTPUT_PATH}/Small3_fullres/fold_{fold}"
    os.makedirs(output_folder)
    #output_folder.mkdir(parents=True, exist_ok=True)

    command_to_run = f"mednextv1_predict -i {str(MODEL_INPUT_PATH)} -o {str(output_folder)} -tr nnUNetTrainerV2_MedNeXt_S_kernel3 -t 505 -f {str(fold)} -p nnUNetPlansv2.1_trgSp_1x1x1 --save_npz"
    result = subprocess.run(command_to_run, shell=True, text=True, capture_output=True)
    print(f"RESULTS MEDNEXT FOLD {fold}")
    print(result)
    print(f"Time for MedNeXt S3 full resolution inference (fold {fold}): {time.time() - start_mednext} seconds")

    """
    subprocess.run(
        "mednextv1_predict",
        "-i", str(MODEL_INPUT_PATH),
        "-o", str(output_folder),
        "-tr", "nnUNetTrainerV2_MedNeXt_S_kernel3",
        "-t", "505",
        "-f", str(fold),
        "-p", "nnUNetPlansv2.1_trgSp_1x1x1",
        "--save_npz"
    , check=True)
    """


def nnunet_fullres_inference():
    start_nn_fullres = time.time()
    output_folder = f"{NNUNET_OUTPUT_PATH}/fullres"
    os.makedirs(output_folder, exist_ok=True)

    # You can either point to a directory containing all fold subfolders or 
    # reference all fold weights at once using nnUNet's capabilities
    command_to_run = [
        "nnUNetv2_predict",
        "-i", str(MODEL_INPUT_PATH),  # Input data folder
        "-o", str(output_folder),  # Output folder
        "-d", "505",  # Dataset identifier
        "-c", "3d_fullres",  # Configuration (3D full resolution)
        "-device", "cuda",  # Use GPU for inference
        "--save_probabilities",  # If you want to save probability maps
        "--disable_tta",  # Disable test time augmentation to make inference faster
        "--verbose"
    ]
    
    # Execute the command once for all folds
    result = subprocess.run(command_to_run)
    print(f"RESULTS FULL RESOLUTION")
    print(result)

    subprocess_time = time.time() - start_nn_fullres
    print(f"Total time for nnUNet full resolution inference: {subprocess_time} seconds")

    """
    subprocess.run(
        "nnUNetv2_predict",
        "-i", str(MODEL_INPUT_PATH),
        "-o", str(output_folder),
        "-d", "505",
        "-c", "3d_fullres",
        "-f", str(fold),
        "--save_probabilities"
    , check=True)
    """


def nnunet_lowres_inference():
    start_nn_lowres = time.time()
    output_folder = f"{NNUNET_OUTPUT_PATH}/lowres"
    os.makedirs(output_folder, exist_ok=True)

    # Run all folds at once for lowres configuration
    command_to_run = [
        "nnUNetv2_predict",
        "-i", str(MODEL_INPUT_PATH),  # Input data folder
        "-o", str(output_folder),  # Output folder
        "-d", "505",  # Dataset identifier
        "-c", "3d_lowres",  # Configuration (3D low resolution)
        "-device", "cuda",  # Use GPU for inference
        "--save_probabilities",  # If you want to save probability maps
        "--disable_tta",  # Disable test time augmentation
        "--verbose"
    ]

    result = subprocess.run(command_to_run)
    print(f"RESULTS LOW RESOLUTION")
    print(result)

    subprocess_time = time.time() - start_nn_lowres
    print(f"Total time for nnUNet low resolution inference: {subprocess_time} seconds")

    """
    subprocess.run(
        "nnUNetv2_predict",
        "-i", str(MODEL_INPUT_PATH),
        "-o", str(output_folder),
        "-d", "505",
        "-c", "3d_lowres",
        "-f", str(fold),
        "--save_probabilities"
    , check=True)
    """


def nnunet_cascade_inference():
    start_nn_cascade = time.time()
    output_folder = f"{NNUNET_OUTPUT_PATH}/cascade"
    os.makedirs(output_folder, exist_ok=True)

    prev_stage_predictions_path = f"{NNUNET_OUTPUT_PATH}/lowres"

    # Run all folds at once for lowres configuration
    command_to_run = [
        "nnUNetv2_predict",
        "-i", str(MODEL_INPUT_PATH),  # Input data folder
        "-o", str(output_folder),  # Output folder
        "-d", "505",  # Dataset identifier
        "-c", "3d_cascade_fullres",  # Configuration (3D low resolution)
        "-prev_stage_predictions", str(prev_stage_predictions_path), 
        "-device", "cuda",  # Use GPU for inference
        "--save_probabilities",  # If you want to save probability maps
        "--disable_tta",  # Disable test time augmentation
        "--verbose"
    ]

    result = subprocess.run(command_to_run)
    print(f"CASCADE RESOLUTION")
    print(result)

    subprocess_time = time.time() - start_nn_cascade
    print(f"Total time for nnUNet cascade resolution inference: {subprocess_time} seconds")

    """
    subprocess.run(
        "nnUNetv2_predict",
        "-i", str(MODEL_INPUT_PATH),
        "-o", str(output_folder),
        "-d", "505",
        "-c", "3d_cascade_fullres",
        "-f", str(fold),
        "-prev_stage_predictions", str(prev_stage_predictions_path),
        "--save_probabilities"
    , check=True)
    """


def nnunet_ensemble():
    start_ensemble = time.time()

    # Directories containing ensembled outputs for fullres, lowres, and cascade
    fullres_dir = f"{NNUNET_OUTPUT_PATH}/fullres"
    lowres_dir = f"{NNUNET_OUTPUT_PATH}/lowres"
    cascade_dir = f"{NNUNET_OUTPUT_PATH}/cascade"  # If you have a cascade model output

    # Prepare the input argument as a list of directories
    input_dirs = [fullres_dir, lowres_dir, cascade_dir]

    # Run the ensemble command on these ensembled outputs
    command_to_run = [
        "nnUNetv2_ensemble",
        "-i", *input_dirs,  # Expand the input_dirs list as separate arguments
        "-o", str(NNUNET_ENSEMBLE_OUTPUT_PATH),
        "--save_npz"
    ]

    # Use subprocess.run with a list of arguments
    result = subprocess.run(command_to_run)

    print("RESULTS FINAL ENSEMBLE")
    print(result)
    print(f"Time for final nnUNet ensemble: {time.time() - start_ensemble} seconds")

    """
    subprocess.run(
        "nnUNetv2_ensemble",
        "-i" + [str(folder) for folder in all_fold_folders] + 
        "-o", str(NNUNET_ENSEMBLE_OUTPUT_PATH),
        "--save_npz"
    , check=True)
    """


def mednext_ensemble():
    start_mednext_ensemble = time.time()
    all_fold_folders = []

    for model in ["Small3_fullres"]:
        fold_folders = [MEDNEXT_OUTPUT_PATH / model / f"fold_{fold}" for fold in range(5)]
        all_fold_folders.extend(fold_folders)

    subprocess.run(
        "mednextv1_ensemble",
        "-f" + [str(folder) for folder in all_fold_folders] +
        "-o", str(MEDNEXT_ENSEMBLE_OUTPUT_PATH),
        "--npz"
    , check=True)
    print(f"Time for MedNeXt ensemble: {time.time() - start_mednext_ensemble} seconds")


def convert_and_save_final_output(input_name):
    start_convert_output = time.time()
    final_output_file = NNUNET_ENSEMBLE_OUTPUT_PATH / "concatenated_500.nii.gz"
    output_mha_file = OUTPUT_PATH / f"{input_name}/mri-head-neck-segmentation/output.mha"
    output_folder = OUTPUT_PATH / f"{input_name}/mri-head-neck-segmentation"
    output_folder.mkdir(parents=True, exist_ok=True)

    image = sitk.ReadImage(str(final_output_file))
    sitk.WriteImage(image, str(output_mha_file))
    print(f"Time to convert and save final output: {time.time() - start_convert_output} seconds")


def _show_torch_cuda_info():
    import torch
    start_torch_info = time.time()
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)
    print(f"Time to collect Torch CUDA information: {time.time() - start_torch_info} seconds")


if __name__ == "__main__":
    raise SystemExit(run())
