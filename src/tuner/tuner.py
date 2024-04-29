import mlflow

class SelectFromRun:
    def __init__(self, experiment_name, metrics_name="metrics.fscore") -> None:
        # retrieves the runs from the experiments
        self.experiment_name = experiment_name
        client = mlflow.MlflowClient()
        current_experiment = dict(client.get_experiment_by_name(experiment_name))
        self.experiment_id = current_experiment['experiment_id']
        self.runs = client.search_runs([self.experiment_id], order_by=[f"{metrics_name} DESC"])
    
    def getBestRun(self, task_name=None, params=None):
        # just a for loop and find the first that match (as its ordered, should select the best fscore)
        if params is None:
            return self.runs[0]
        
        if task_name is not None:
            for run in self.runs:
                if run.data.task_name == task_name:
                    return run
        
        for run in self.runs:
            if run.data.params == {k:str(v) for k, v in params.items()}:
                return run
        return None
