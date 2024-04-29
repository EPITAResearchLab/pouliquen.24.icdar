
python train.py --config-name=classifier -m +experiment=classifier/mobilevit_s "paths.split_name=k0,k1,k2,k3,k4"
python calibration.py --config-name=classifier -m +experiment=classifier/mobilevit_s "paths.split_name=k0,k1,k2,k3,k4"
python test.py --config-name=classifier -m +experiment=classifier/mobilevit_s "paths.split_name=k0,k1,k2,k3,k4"
