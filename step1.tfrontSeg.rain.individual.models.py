import os
import subprocess
from datetime import datetime
import torch

print(os.getcwd())
# os.chdir('mmsegmentation')

os.getcwd()

if __name__ == '__main__':
    for fold in range(1, 5):
        print(f"===this is in fold-{fold}===")

        # for model in ["UNet", "FastSCNN", "Segformer", "PSPNet", "KNet", "DeepLabV3plus"]:
        for model in ["DeepLabV3plus"]:
            print(f"\t**this is in fold-{fold} of model {model}**")
            configFile="Project-Configs_frontSeg_2024apr\Front_Fold"+str(fold)+"_configs\Fold"+str(fold)+"_"+model+".py"
            print(f"\t["+configFile+"]")

            cmdStr = r"python tools\train.py "+configFile
            print(f"\t["+cmdStr+"]")

            # Add this at the very top of your script, before any other torch operations
            # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1

            # Check if GPU is available
            device = torch.device(
                "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            if device.type == 'cuda':
                print(f"Available GPU count: {torch.cuda.device_count()}")
                print(f"GPU Name: {torch.cuda.get_device_name(device.index)}")
                print(f"Memory Allocated: {torch.cuda.memory_allocated(device.index) / 1024 ** 2:.2f} MB")

            # Add after device selection
            if torch.cuda.is_available():
                print(f"Current device index: {torch.cuda.current_device()}")
                for i in range(torch.cuda.device_count()):
                    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
                    print(f"Memory allocated on device {i}: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")

            subprocess.run(cmdStr, shell=True, check=True)
