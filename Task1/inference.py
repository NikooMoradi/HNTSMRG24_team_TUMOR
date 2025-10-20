"""
The following is a simple example algorithm for Task 2 (mid-RT segmentation) of the HNTS-MRG 2024 challenge.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm-task-2-mid-rt-segmentation | gzip -c > example-algorithm-task-2-mid-rt-segmentation.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""

from pathlib import Path
from glob import glob
import SimpleITK as sitk
import subprocess
import os
import torch
import time 

INPUT_PATH = Path("/input")  # For Docker
OUTPUT_PATH = Path("/output")  #

MODEL_INPUT_PATH = OUTPUT_PATH / "model_input"  
MEDNEXT_OUTPUT_PATH = OUTPUT_PATH / "mednext_output"  


def run():
    start_time = time.time()
    print("Start time:", time.ctime(start_time))

    """
    Main function to read input, process the data, and write output.
    """
    os.environ['RESULTS_FOLDER'] = 'models/RESULTS_FOLDER/'
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
        start_mednex = time.time()
        mednext_S3_fullres_inference()  # Now handles all folds
        print(f"Time for mednext inference: {time.time() - start_mednex} seconds")

        # Convert and save the final output (remains the same)
        start_convert_output = time.time()
        convert_and_save_final_output(input_dir.name)
        print(f"Time to convert and save final output: {time.time() - start_convert_output} seconds")

    total_time = time.time() - start_time
    print("Total execution time:", total_time, "seconds")
    return 0


def convert_and_rename_inputs(input_dir):
    MODEL_INPUT_PATH.mkdir(parents=True, exist_ok=True)

    name_mapping = {
        "pre-rt-t2w-head-neck": "preRT_500_0000.nii.gz",
    }

    for dir_name, output_filename in name_mapping.items():
        input_folder = input_dir / dir_name
        input_file = glob(str(input_folder / "*.mha"))[0]  # Get the .mha file inside the folder
        output_file = MODEL_INPUT_PATH / output_filename
        convert_mha_to_niigz(input_file, output_file)


def convert_mha_to_niigz(input_file, output_file):
    image = sitk.ReadImage(str(input_file))
    sitk.WriteImage(image, str(output_file))

def mednext_S3_fullres_inference():
    start_mednext = time.time()    
    os.makedirs(MEDNEXT_OUTPUT_PATH, exist_ok=True)

    command_to_run = [
        "mednextv1_predict",
        "-i", str(MODEL_INPUT_PATH),
        "-o", str(MEDNEXT_OUTPUT_PATH),
        "-tr", "nnUNetTrainerV2_MedNeXt_S_kernel3",
        "-t", "501",
        "-p", "nnUNetPlansv2.1_trgSp_1x1x1",
        "--save_npz"
        ]

    # Execute the command once for all folds
    result = subprocess.run(command_to_run)
    print(f"RESULTS FULL RESOLUTION")
    print(result)

    subprocess_time = time.time() - start_mednext
    print(f"Total time for mednext inference: {subprocess_time} seconds")


def convert_and_save_final_output(input_name):
    print("list dir MEDNEXT_OUTPUT_PATH")
    print(os.listdir(MEDNEXT_OUTPUT_PATH))
    final_output_file = MEDNEXT_OUTPUT_PATH / "preRT_500.nii.gz"
    output_mha_file = OUTPUT_PATH / f"{input_name}/mri-head-neck-segmentation/output.mha"
    output_folder = OUTPUT_PATH / f"{input_name}/mri-head-neck-segmentation"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Convert the .nii.gz output to .mha format
    image = sitk.ReadImage(str(final_output_file))
    sitk.WriteImage(image, str(output_mha_file))

def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
