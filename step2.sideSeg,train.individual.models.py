import os
import subprocess
from datetime import datetime

print(os.getcwd())
# os.chdir('mmsegmentation')

os.getcwd()

if __name__ == '__main__':
    for fold in range(1, 5):
        print(f"===this is in fold-{fold}===")

        for model in ["unet", "FastSCNN", "segformer", "PSPNet", "knet", "deepLabV3plus"]:
            print(f"\t**this is in fold-{fold} of model {model}**")
            configFile=r"projectConfig_sideSeg_2024Oct30\fold_"+str(fold)+"_"+model+".py"
            print(f"\t["+configFile+"]")

            cmdStr = r"python tools\train.py "+configFile + " --gpu-id 1"
            print(f"\t["+cmdStr+"]")

            subprocess.run(cmdStr, shell=True, check=True)
