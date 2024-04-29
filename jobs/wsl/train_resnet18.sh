python train.py --config-name=wsl -m +experiment=wsl/resnet18 "paths.split_name=k0,k1,k2,k3,k4"

python calibration.py --config-name=wsl -m +experiment=wsl/resnet18 "paths.split_name=k0,k1,k2,k3,k4"

python test.py --config-name=wsl -m +experiment=wsl/resnet18 "paths.split_name=k0,k1,k2,k3,k4"