echo "training model"
python train.py --config-name=wsl -m +experiment=wsl/mobilevit_s "paths.split_name=k0,k1,k2,k3,k4"

echo "all video"
python calibration.py --config-name=wsl -m +experiment=wsl/mobilevit_s "paths.split_name=k0,k1,k2,k3,k4" "decision=allvideo"

python test.py --config-name=wsl -m +experiment=wsl/mobilevit_s "paths.split_name=k0,k1,k2,k3,k4" "decision=allvideo"

echo "cumulative"
python calibration.py --config-name=wsl -m +experiment=wsl/mobilevit_s "paths.split_name=k0,k1,k2,k3,k4" "decision=cumulative" 
python test.py --config-name=wsl -m +experiment=wsl/mobilevit_s "paths.split_name=k0,k1,k2,k3,k4" "decision=cumulative"
