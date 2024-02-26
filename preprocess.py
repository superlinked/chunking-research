from abc import abstractmethod
from pathlib import Path
from typing import Union

from datasets import load_dataset
from hydra import compose, initialize
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm

from utils import get_logger


class Dataset:

    columns = ['context', 'questions', 'answers']

    @staticmethod
    @abstractmethod
    def load_dataset(config: DictConfig) -> pd.DataFrame:
        pass


class QUAC(Dataset):
    # https://quac.ai/
    # https://huggingface.co/datasets/quac?library=true
    @staticmethod
    def load_dataset(config: DictConfig) -> pd.DataFrame:

        logger = get_logger()
        dataset_name = 'quac'
        path_cache_dir = Path().resolve().joinpath(
            config.datasets, dataset_name
        )
        logger.info(f'Loading dataset: {dataset_name}')
        dataset = load_dataset(dataset_name, cache_dir=str(path_cache_dir))
        df_train = pd.DataFrame(dataset['train'])
        df_valid = pd.DataFrame(dataset['validation'])

        if config.preprocess.concat_train_validation:
            logger.info('Concatenating train and validation sets...')
            df = pd.concat([df_train, df_valid], axis=0)
        else:
            df = df_train.copy()

        curated_samples = []
        samples = df[['context', 'questions', 'answers']].to_dict('records')

        for sample in tqdm(samples):
            curated_samples.append({
                'context': sample['context'],
                'questions': sample['questions'],
                'answers': [answer[0] for answer in sample['answers']['texts']]
            })

        df = pd.DataFrame(curated_samples)

        # There are separate and therefore duplicated contexts with different
        # questions and answers - aggregate them
        df = df.groupby('context').agg({
            'questions': lambda x: [item for sublist in x for item in sublist],
            'answers': lambda x: [item for sublist in x for item in sublist]
        }).reset_index()
        # Remove corresponding questions and answers where the answer is
        # 'CANNOTANSWER'
        questions = [
            [qu for qu, ans in zip(questions, answers)
             if ans != 'CANNOTANSWER'] for questions, answers in
            zip(df['questions'], df['answers'])
        ]
        answers = [
            [ans for ans in answers if ans != 'CANNOTANSWER']
            for answers in df['answers']
        ]
        df['questions'], df['answers'] = questions, answers

        df = df.loc[:, QUAC.columns]

        n_contexts = config.preprocess.n_contexts

        if n_contexts:
            df = df.iloc[:n_contexts]

        return df


class SQUAD(Dataset):
    # https://rajpurkar.github.io/SQuAD-explorer/
    # https://huggingface.co/datasets/squad?row=0
    @staticmethod
    def load_dataset(config: DictConfig) -> pd.DataFrame:

        logger = get_logger()
        dataset_name = 'squad'
        path_cache_dir = Path().resolve().joinpath(
            config.datasets, dataset_name
        )
        logger.info(f'Loading dataset: {dataset_name}')
        dataset = load_dataset(dataset_name, cache_dir=str(path_cache_dir))
        df_train = pd.DataFrame(dataset['train'])
        df_valid = pd.DataFrame(dataset['validation'])

        if config.preprocess.concat_train_validation:
            logger.info('Concatenating train and validation sets...')
            df = pd.concat([df_train, df_valid], axis=0)
        else:
            df = df_train.copy()

        df = df.loc[:, ['context', 'question', 'answers']].copy()
        # although it is called answers, there is always one element only
        df['answers'] = df.answers.apply(lambda x: x['text'][0])

        df.rename(
            columns={'question': 'questions', 'answer': 'answers'},
            inplace=True
        )
        df = df.loc[:, SQUAD.columns]

        df = df.groupby('context').agg(
            {'questions': list, 'answers': list}
        ).reset_index()

        n_contexts = config.preprocess.n_contexts

        if n_contexts:
            df = df.iloc[:n_contexts]

        return df


def factory(
    config: DictConfig
) -> Union[QUAC, SQUAD]:

    return globals()[config.preprocess.dataset]


if __name__ == '__main__':

    initialize(config_path='configs', version_base='1.1')
    config = compose(config_name='config')

    print(config)

    df = QUAC.load_dataset(config)
    print(df.head())

    df = SQUAD.load_dataset(config)
    print(df.head())
