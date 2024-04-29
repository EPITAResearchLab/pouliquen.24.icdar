time python3 calibration.py --config-name=midv_baseline -m "+experiment=joblib" "paths.split_name=k0,k1,k2,k3,k4" +"tune=True" "model.T=40" "model.s_t=30" "decision.th=0.01" "decision=allvideo"
time python3 calibration.py --config-name=midv_baseline -m "+experiment=joblib" "paths.split_name=k0,k1,k2,k3,k4" +"tune=True"  "model.T=range(10,200,10)" "model.s_t=30,40,50" "decision.th=0" "decision=allvideo"

time python3 test.py --config-name=midv_baseline -m "+experiment=joblib" "paths.split_name=k0,k1,k2,k3,k4" "decision.th=0" +"tune=True" "decision=allvideo"
