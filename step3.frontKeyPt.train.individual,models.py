import os
import subprocess
from datetime import datetime

print(os.getcwd())
os.chdir('mmpose')

os.getcwd()

if __name__ == '__main__':
    for fold in range(1, 5):
        print(f"===this is in fold-{fold}===")
        configPrefix=dict(mobilenetv2="td-hm_mobilenetv2_",
            hrnet="Hrnet_w32_",
            hourglass52="td-hm_hourglass52_",
            rtmpose="rtmpose-s_",
            shufflenet ="td-hm-shufflenetv1_")
        configSuffix=dict(hourglass52 ="_front_coco-256x256",
                          hrnet="_front_coco-256x192", rtmpose="_front_coco-256x192", shufflenet="_front_coco-256x192",mobilenetv2="_front_coco-256x192")
        thisFold="Fold"+str(fold)
        for model in ["rtmpose"]: #["shufflenet", "hourglass52", "hrnet", "mobilenetv2"]: #, "rtmpose"]:
            print(f"\t**this is in fold-{fold} of model {model}**")
            configFile=os.path.join(r"configs_dec17_front_keypt", "Front_configs_"+model, configPrefix[model] + thisFold + configSuffix[model] + ".py") #.replace("\\", "\\\\")
            #
           # print(rf"[[{configFile}]]")
            print(os.path.exists(rf"{configFile}"))

            cmdStr = rf"python tools\train.py {configFile}  --work-dir work_dirs_Front_dec18\{model}_fold{fold}"
            print(f"\t["+cmdStr+"]")

            subprocess.run(cmdStr, shell=True, check=True)
            # break