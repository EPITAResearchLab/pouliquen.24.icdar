
python train.py --config-name=wsl -m +experiment=wsl/resnet18_noaugment "paths.split_name=k0,k1,k2,k3,k4"

python calibration.py --config-name=wsl -m +experiment=wsl/resnet18_noaugment "paths.split_name=k0,k1,k2,k3,k4" "decision=allvideo"
python test.py --config-name=wsl -m +experiment=wsl/resnet18_noaugment "paths.split_name=k0,k1,k2,k3,k4" "decision=allvideo"