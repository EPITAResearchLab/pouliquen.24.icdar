import mlflow
from tqdm import tqdm
import ast

def already_run(cfg, experiment_name):
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name) # get all experiments
    if experiment is not None:
        runs = client.search_runs([experiment.experiment_id]) # get all runs from an experiment
        for run in runs:
            if run.data.params == {k:str(v) for k, v in cfg.items()} and "error_message" not in run.data:
                return True
    return False

def values_as_list(d, exclude_k="_target_"):
    return [f"{k}{v}" for k, v in d.items() if k != exclude_k]

def get_metrics_from_values(acc_origins, acc_frauds):
    # acc_frauds is what was predicted, 1 means that its a tp
    # calculating stats
    origin_fp = acc_origins
    frauds_tp = acc_frauds

    tp = sum(frauds_tp)
    fp = sum(origin_fp)
    fn = len(frauds_tp) - tp
    tn = len(origin_fp) - fp
    if len(acc_frauds) == 0: # only origins
        return {"specificity":1-sum(acc_origins)/len(acc_origins)}
    elif len(acc_origins) == 0: # only frauds
         return {"recall":sum(acc_frauds)/len(acc_frauds)}
    
    # fscore = 2*tp/(2*tp+fp+fn)
    sumof = sum(acc_origins)+sum(acc_frauds)
    if sumof:
        precision = sum(acc_frauds) / sumof
        recall = sum(acc_frauds) / len(acc_frauds)
        if not precision or not recall:
            fscore = 0
        else:
            fscore = 2*precision*recall / (precision+recall)
    else:
        precision = 0
        recall = 0
        fscore = 0
    metrics = {"fscore":fscore, "recall":recall, "precision":precision, "specificity":1-sum(acc_origins)/len(acc_origins), "fp": fp, "tp":tp, "fn":fn, "tn":tn}
    return metrics

def get_metrics(dataset, model, decision):
    # the predictions
    acc_frauds = []
    acc_origins = []
    
    for i in tqdm(range(len(dataset))):
        model.reset()
        frames = dataset[i]
        d, j = decision.process_frame_by_frame(frames, model)
        if dataset.isFraud(i):
            acc_frauds.append(d)
        else:
            acc_origins.append(d)
    
    metrics = get_metrics_from_values(acc_origins, acc_frauds)
    
    return metrics

def get_fscore(thr, origins_t, frauds_t):
    origin_fp = origins_t < thr
    frauds_tp = frauds_t < thr

    return get_metrics_from_values(origin_fp, frauds_tp)["fscore"]

def mlruntodict(mlparams):
    return {k:(ast.literal_eval(v) if "'" in v else v) for k,v in mlparams.items()}

def get_best_run(experiment_name, metrics_name, task_name=None, run_name=None, params=None):
    # retrieves the runs from the experiments
    experiment_name = experiment_name
    client = mlflow.MlflowClient()
    current_experiment = dict(client.get_experiment_by_name(experiment_name))
    experiment_id = current_experiment['experiment_id']
    runs = client.search_runs([experiment_id], order_by=[f"metrics.{metrics_name} ASC"])

    if params is None and task_name is None:
        return runs[0]
    
    if task_name is not None:
        for run in runs:
            if task_name == run.data.tags.get("task_name", "") and (run_name != None and run.info.run_name == run_name):
                return run
    if params is not None:
        for run in runs:
            if run.data.params == {k:str(v) for k, v in params.items()}:
                return run
    return None