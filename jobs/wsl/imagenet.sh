python calibration.py --config-name=imagenet -m "paths.split_name=k0,k1,k2,k3,k4" "decision=allvideo" model.model.backbone_name=mobilevit_xxs,mobilenetv3_small_050,resnet18 
python test.py --config-name=imagenet -m "paths.split_name=k0,k1,k2,k3,k4" "decision=allvideo" model.model.backbone_name=mobilevit_xxs,mobilenetv3_small_050,resnet18
