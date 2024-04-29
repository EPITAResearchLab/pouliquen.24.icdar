import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
import logging
import mlflow
from src.utils.utils import already_run, get_best_run, values_as_list, get_metrics
from os.path import join as pjoin
from lightning import seed_everything

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf")
def main(cfg : DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)

    data_val = instantiate(cfg.data.train)

    decision = instantiate(cfg.decision)
    # If training has been conducted, the optimal model with regard to the loss will be selected
    if cfg.get("training", ""):
        print(cfg.training)
        run_name = cfg.training.trainer.get('run_name', None)
        run_name = f"{cfg.task_name}_{run_name}" if run_name is not None else None
        # log.info(cfg.task_name + " run name: " + run_name)
        run = get_best_run("lightning_logs", "best_val_loss", cfg.task_name, run_name=run_name)
        if run is not None:
            path = os.path.join(os.path.dirname(run.info.artifact_uri.replace("file://", "")), "checkpoints")
            model_name = next((p for p in os.listdir(path) if p.startswith("backbone_")), "")
            if len(model_name) > 0: 
                with open_dict(cfg):
                    cfg.model.model.model_path = pjoin(path, model_name)
                log.info(f"found the model {model_name}")
            else:
                log.error("couldnt find the network")
                return
        else:
            log.error("couldnt find the run for the network")
            return
        log.info(cfg.model)
    
    log.info(cfg.model)
    model = instantiate(cfg.model)
    task_name = cfg.task_name
    
    task_name_full = HydraConfig.get().runtime.choices.decision + "_" + task_name
    if cfg.get("training", ""):
        task_name_full += cfg.training.trainer.run_name
    mlflow.set_experiment(task_name_full)
    if not already_run(cfg, task_name_full):
        if cfg.get("tune", "") == True:
            log.info("tunning parameters")
            metrics, th = decision.tune(data_val, model)
            log.info(f"best fscore found {metrics['fscore']} for th {th}")
            cfg.decision.th = float(th)
        else:
            log.info(f"running for current parameters {cfg.decision.th}")
            metrics = get_metrics(data_val, model, decision)
            log.info(f"fscore of {metrics['fscore']}") 
        with mlflow.start_run():
            mlflow.set_tag("mlflow.runName", f"{cfg.get('task_name', '')}_{'_'.join(values_as_list(cfg.model))}_{'_'.join(values_as_list(cfg.decision))}")
            # tune model
            mlflow.log_params(cfg)
            mlflow.log_metrics(metrics)
    else:
        log.info("this run was already lauched")

if __name__ == "__main__":
    main()