import inspect
import logging
from pathlib import Path
from typing import Dict, List

import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import CrossEncoder, SentenceTransformer

file_query_embeddings = 'query_embeddings.npy'


def get_logger() -> logging.Logger:
    caller = inspect.stack()[1][3]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(caller)


def save_query_embeddings(
    config: DictConfig,
    query_embeddings: np.ndarray
) -> None:

    path_embeddings = get_path_of_query_embeddings(config)

    if not path_embeddings.is_dir():
        path_embeddings.parent.mkdir(parents=True)

    with open(path_embeddings, 'wb') as file:
        np.save(file, query_embeddings)


def load_query_embeddings(
    config: DictConfig,
    questions: List[str]
) -> np.ndarray:

    path_embeddings = get_path_of_query_embeddings(config)

    with open(path_embeddings, 'rb') as file:
        query_embeddings = np.load(file)

    assert len(questions) == len(query_embeddings)

    return query_embeddings


def check_if_query_embeddings_exist(config) -> bool:
    path_embeddings = get_path_of_query_embeddings(config)

    return True if path_embeddings.is_file() else False


def get_path_of_query_embeddings(config: DictConfig) -> Path:

    path_embeddings = Path().resolve().joinpath(
        config.embeddings,
        config.preprocess.dataset,
        config.retrieval.model.model_name,
        file_query_embeddings
    )
    return path_embeddings


def log_retrieval_experiment_mlflow(
    config: DictConfig,
    top_k: int,
    metrics: Dict[str, float],
    keys_to_remove: List[str]
) -> None:

    logger = get_logger()
    logger.info('Logging results with MLFlow...')
    config = select_relevant_config(config, keys_to_remove)
    cr = config.retrieval
    cfg_mlflow = config.mlflow
    mlflow.set_tracking_uri(Path().resolve().joinpath(config.experiments))
    mlflow.set_experiment(cfg_mlflow.experiment_name)

    # llama_index HuggingfaceEmbedding doesn't store number of parameters...
    model_name = cr.model.model_name
    model = SentenceTransformer(model_name, device='cpu')
    params_to_log = {}

    n_params = sum(p.numel() for p in model.parameters())
    params_to_log.update({
        'embedding_model_params': f'{n_params:,}',
        'embedding_model_name': model_name,
        'top_k': top_k,
        'chunker': cr.chunker.name,
        **cr.chunker.params
    })
    if cr.reranking:
        model_name = cr.reranker.model_name
        model = CrossEncoder(model_name, device='cpu')
        n_params = sum(p.numel() for p in model.model.parameters())
        params_to_log.update({
            'reranker': model_name, 'reranker_params': f'{n_params:,}'
        })

    tags = {
        **cfg_mlflow.tags,
        'chunker': cr.chunker.name,
        'dataset': config.preprocess.dataset
    }
    with mlflow.start_run(
        run_name=cfg_mlflow.run_name,
        description=cfg_mlflow.description
    ) as run:

        mlflow.set_tags(tags)
        mlflow.log_dict(OmegaConf.to_container(config), 'config.yaml')
        mlflow.log_metrics(metrics)
        mlflow.log_params(params_to_log)


def log_colbert_experiment_mlflow(
    config: DictConfig,
    top_k: int,
    metrics: Dict[str, float],
    keys_to_remove: List[str]
) -> None:

    logger = get_logger()
    logger.info('Logging results with MLFlow...')
    config = select_relevant_config(config, keys_to_remove)
    ccr = config.colbert_retrieval

    cfg_mlflow = config.mlflow
    mlflow.set_tracking_uri(Path().resolve().joinpath(config.experiments))
    mlflow.set_experiment(cfg_mlflow.experiment_name)

    params_to_log = {
        'model_name': ccr.model_name,
        'max_document_length': ccr.max_document_length,
        'top_k': top_k
    }
    tags = {
        **cfg_mlflow.tags,
        'dataset': config.preprocess.dataset
    }
    with mlflow.start_run(
        run_name=cfg_mlflow.run_name,
        description=cfg_mlflow.description
    ) as run:

        mlflow.set_tags(tags)
        mlflow.log_dict(OmegaConf.to_container(config), 'config.yaml')
        mlflow.log_metrics(metrics)
        mlflow.log_params(params_to_log)


def select_relevant_config(
    config: DictConfig,
    keys_to_remove: List[str]
) -> DictConfig:
    # remove irrelevant part related to an experiment
    config._set_flag("struct", False)

    if 'pipeline' in config.keys():
        del config['pipeline']

    for key in keys_to_remove:
        if key in config.keys():
            del config[key]

    config._set_flag("struct", True)

    return config


def recursive_config_update(
    cfg_original: dict,
    cfg_update: dict
) -> dict:
    """"""
    for key, value in cfg_update.items():
        if (
            isinstance(value, dict)
            and key in cfg_original
            and isinstance(cfg_original[key], dict)
        ):
            recursive_config_update(cfg_original[key], value)
        else:
            cfg_original[key] = value
