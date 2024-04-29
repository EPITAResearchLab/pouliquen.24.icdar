import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything

@hydra.main(version_base=None, config_path="conf")
def main(cfg : DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)
    print(OmegaConf.to_yaml(cfg))

    trainer = instantiate(cfg.training.trainer)

    datamodule = instantiate(cfg.training.datamodule)

    trainer.train(datamodule, cfg.task_name)
    
if __name__ == "__main__":
    main()