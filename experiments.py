from abc import ABC, abstractmethod

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from retrieval import (
    run_retrieval_evaluation,
    run_colbert_retrieval_evaluation
)
from utils import (
    get_logger,
    log_retrieval_experiment_mlflow,
    log_colbert_experiment_mlflow,
    recursive_config_update
)


class ExperimentRunner(ABC):

    @staticmethod
    @abstractmethod
    def run_experiment(cfg: DictConfig):
        pass


class Retrieval(ExperimentRunner):
    @staticmethod
    def run_experiment(cfg: DictConfig):
        results = run_retrieval_evaluation(cfg)
        top_ks = cfg.retrieval.evaluation.top_ks

        for idx, top_k in enumerate(top_ks):
            log_retrieval_experiment_mlflow(
                cfg, top_k, results[idx], ['colbert_retrieval']
            )


class RetrievalColBERT(ExperimentRunner):
    @staticmethod
    def run_experiment(cfg: DictConfig):
        results = run_colbert_retrieval_evaluation(cfg)
        top_ks = cfg.retrieval.evaluation.top_ks

        for idx, top_k in enumerate(top_ks):
            log_colbert_experiment_mlflow(
                cfg, top_k, results[idx], ['retrieval']
            )


@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def main(cfg: DictConfig) -> None:
    def wrap_experiment(cfg: DictConfig, exp_name: str) -> None:

        if exp_name in globals() and callable(globals()[exp_name]):
            experiment_runner: ExperimentRunner = globals()[exp_name]
            experiment_runner.run_experiment(cfg)
        else:
            raise NotImplementedError(
                f'Configured experiment class {exp_name} is not found!'
            )

    logger = get_logger()

    if cfg.pipeline:
        logger.info(f'Running {len(cfg.pipeline)} experiments serially.')

        for exp_params in tqdm(cfg.pipeline):
            cfg_exp = cfg.copy()
            dict_exp = OmegaConf.to_container(cfg_exp, resolve=True)
            dict_update = OmegaConf.to_container(exp_params, resolve=True)
            recursive_config_update(dict_exp, dict_update)
            cfg_exp = OmegaConf.create(dict_exp)
            exp_name = cfg_exp.mlflow.experiment_name
            logger.info(
                f'Running {exp_name} experiment with parameters: {exp_params}'
            )
            wrap_experiment(cfg_exp, exp_name)
    else:
        logger.info('Running experiment with the main config.')
        exp_name = cfg.mlflow.experiment_name
        wrap_experiment(cfg, exp_name)


if __name__ == '__main__':

    main()
