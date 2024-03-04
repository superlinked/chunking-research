import time
from pathlib import Path
from typing import Dict, List, Union

from omegaconf import DictConfig
import faiss
from faiss.swigfaiss import IndexIDMap
from hydra import compose, initialize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import (
    LangchainNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
    SemanticSplitterNodeParser,
    TokenTextSplitter
)
import numpy as np
from ragatouille import RAGPretrainedModel
from sentence_transformers import CrossEncoder
import torch
from transformers import AutoTokenizer
import torchmetrics
from tqdm import tqdm

from preprocess import factory
from utils import (
    get_logger,
    save_query_embeddings,
    load_query_embeddings,
    check_if_query_embeddings_exist,
)


def run_retrieval_evaluation(config: DictConfig) -> List[Dict[str, float]]:

    cr = config.retrieval
    logger = get_logger()

    df = factory(config).load_dataset(config)

    contexts, question_batches = df.context.tolist(), df.questions.tolist()
    # question - doc_id, for retrieval labels
    labels = np.array(
        [idx for idx, qb in enumerate(question_batches) for q in qb]
    )
    questions = [q for qb in question_batches for q in qb]

    logger.info(f'{len(questions):,} questions')

    vs_index = create_vector_store_index(cr, contexts)

    embed_model = vs_index._embed_model
    embedding_dict = vs_index.vector_store.to_dict()

    embeddings = np.array(list(embedding_dict['embedding_dict'].values()))
    node_ids = list(embedding_dict['embedding_dict'].keys())
    node_id_doc_id_map = {
        k: int(v) for k, v in embedding_dict['text_id_to_ref_doc_id'].items()
    }
    vec_node_id_doc_id_mapper = np.vectorize(
        lambda x: node_id_doc_id_map[node_ids[x]]
    )
    faiss_index = create_faiss_index(embeddings, node_ids)

    logger.info(f'Computing {len(questions):,} query embeddings...')

    if (
            check_if_query_embeddings_exist(config)
            and not config.preprocess.n_contexts
    ):
        logger.info('Loading pre-computed query embeddings...')
        query_embeddings = load_query_embeddings(config, questions)
    else:
        query_embeddings = embed_model.get_text_embedding_batch(
            questions, show_progress=True
        )
        query_embeddings = np.array(query_embeddings)

        if not config.preprocess.n_contexts:
            logger.info('Saving query embeddings...')
            save_query_embeddings(config, query_embeddings)

    del embed_model
    results = []

    top_ks = cr.evaluation.top_ks
    max_top_k = max(top_ks)
    # only need to query once with the max of top_ks
    _, retrieved_ids = faiss_index.search(query_embeddings, k=max_top_k)

    if cr.reranking and max_top_k > 1:
        # similarity scores should be computed once with max top_k
        similarity_scores = compute_similarity_scores_for_reranking(
            cr, vs_index, retrieved_ids, questions
        )
        logger.info('Starting evaluation...')
        start = time.time()

        for tk in tqdm(top_ks):
            top_k_retrieved_ids = retrieved_ids[:, :tk].copy()
            top_k_similarity_scores = similarity_scores[:, :tk].copy()
            sorted_indices = np.argsort(top_k_similarity_scores, axis=1)[:, ::-1]

            # rerank retrieved ids based on the similarity scores
            top_k_retrieved_ids = np.take_along_axis(
                top_k_retrieved_ids, sorted_indices, axis=1
            )
            top_k_retrieved_ids = vec_node_id_doc_id_mapper(
                top_k_retrieved_ids
            )
            res_top_k = evaluate_retrieved_results(
                top_k_retrieved_ids, labels, tk
            )
            results.append(res_top_k)
    else:
        logger.info('Starting evaluation...')
        start = time.time()

        for tk in tqdm(top_ks):
            top_k_retrieved_ids = retrieved_ids[:, :tk].copy()
            top_k_retrieved_ids = vec_node_id_doc_id_mapper(
                top_k_retrieved_ids
            )
            res_top_k = evaluate_retrieved_results(top_k_retrieved_ids, labels, tk)
            results.append(res_top_k)

    logger.info(f'Evaluation took {(time.time() - start) / 60:.2f} minutes...')

    return results


def get_document_chunker(
    config: DictConfig,
    embed_model: HuggingFaceEmbedding
) -> Union[
    SentenceSplitter,
    SemanticSplitterNodeParser,
    TokenTextSplitter,
    RecursiveCharacterTextSplitter
]:
    """"""
    conf_chunker = config.chunker
    chunker_name = conf_chunker.name

    if chunker_name in [
        SentenceSplitter.__name__, SentenceWindowNodeParser.__name__
    ]:
        chunker = globals()[chunker_name](**conf_chunker.params)

    elif chunker_name == SemanticSplitterNodeParser.__name__:
        chunker = SemanticSplitterNodeParser(
            embed_model=embed_model, **conf_chunker.params
        )
    elif chunker_name == TokenTextSplitter.__name__:
        tokenizer = AutoTokenizer.from_pretrained(embed_model.model_name)
        chunker = TokenTextSplitter(tokenizer=tokenizer, **conf_chunker.params)
    elif chunker_name == RecursiveCharacterTextSplitter.__name__:
        chunker = LangchainNodeParser(RecursiveCharacterTextSplitter(
            **conf_chunker.params
        ))
    else:
        raise ValueError(f'Unrecognized chunker: {chunker_name}')

    return chunker


def create_vector_store_index(
    config: DictConfig,
    contexts: List[str]
) -> VectorStoreIndex:

    logger = get_logger()
    embed_model = HuggingFaceEmbedding(**config.model)
    documents = [
        Document(
            text=text, doc_id=idx, metadata={"context_id": idx},
            excluded_embed_metadata_keys=["context_id"]
        ) for idx, text in enumerate(contexts)
    ]
    chunker = get_document_chunker(config, embed_model)

    if any([
        isinstance(chunker, x) for x in [
            SentenceSplitter, TokenTextSplitter,
            LangchainNodeParser, SentenceWindowNodeParser]
    ]):
        logger.info(f'Embedding {len(contexts):,} contexts...')
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[chunker],
            embed_model=embed_model,
            show_progress=True
        )
    elif isinstance(chunker, SemanticSplitterNodeParser):
        logger.info(f'Building nodes with {SemanticSplitterNodeParser.__name__}')
        nodes = chunker.build_semantic_nodes_from_documents(
            documents, show_progress=False
        )
        # nodes needs to be filtered as sometimes SemanticSplitterNodeParser
        # produces nodes with empty text...
        nodes = list(filter(lambda x: x.text, nodes))
        logger.info(f'Embedding {len(contexts):,} contexts...')
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embed_model,
            show_progress=True
        )
    else:
        raise ValueError(f'Unrecognized chunker: {chunker}')

    return index


def create_faiss_index(
    embeddings: np.ndarray,
    node_ids: List[str]
) -> IndexIDMap:

    # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dimension))
    index.add_with_ids(embeddings, np.arange(len(node_ids)))

    return index


def compute_similarity_scores_for_reranking(
    config_retrieval: DictConfig,
    index: VectorStoreIndex,
    retrieved_ids: np.ndarray,
    questions: List[str]
) -> np.ndarray:
    """"""
    logger = get_logger()
    # workers are not released in a multiprocessing context
    reranker_name = config_retrieval.reranker.model_name
    model = CrossEncoder(reranker_name, max_length=512)
    logger.info(f'Reranking Retrieval results with {reranker_name}...')

    nodes = list(index.docstore.docs.values())
    vectorized_mapper = np.vectorize(lambda x: nodes[x].text)
    retrieved_chunks = vectorized_mapper(retrieved_ids)

    retrieved_chunks_to_rerank = [
        [qu, chunk] for idx, qu in enumerate(questions)
        for chunk in retrieved_chunks[idx]
    ]
    scores = model.predict(
        retrieved_chunks_to_rerank, **config_retrieval.reranker.predict
    )
    scores = scores.reshape((len(questions), -1))

    return scores


def run_colbert_retrieval_evaluation(
    config: DictConfig
) -> List[Dict[str, float]]:

    ccr = config.colbert_retrieval
    logger = get_logger()
    index_root = Path().resolve().joinpath(config.colbert_index)
    question_batch_size = ccr.question_batch_size
    top_ks = ccr.evaluation.top_ks

    df = factory(config).load_dataset(config)

    contexts, question_batches = df.context.tolist(), df.questions.tolist()
    # question - doc_id, for retrieval labels
    labels = np.array(
        [idx for idx, qb in enumerate(question_batches) for q in qb]
    )
    questions = [q for qb in question_batches for q in qb]

    rag = RAGPretrainedModel.from_pretrained(
        ccr.model_name,
        index_root=str(index_root),
        verbose=1
    )
    start = time.time()
    rag.index(
        collection=contexts,
        document_ids=[str(i) for i in range(len(contexts))],
        max_document_length=ccr.max_document_length,
        split_documents=True
    )
    logger.info(
        f'Indexing took {(time.time() - start) / 60:.2f} minutes.\n'
        f'Starting retrieval search...'
    )
    retrieved_ids = []
    max_top_k = max(top_ks)

    for idx in tqdm(list(range(0, len(questions), question_batch_size))):

        question_batch = questions[idx: idx + question_batch_size]
        retrievals = rag.search(query=question_batch, k=max_top_k)
        retrieved_ids_batch = np.array([
            int(x['document_id']) for sub_list in retrievals for x in sub_list
        ]).reshape((-1, max_top_k))
        retrieved_ids.append(retrieved_ids_batch)

    retrieved_ids = np.concatenate(retrieved_ids, axis=0)
    results = []
    logger.info('Starting evaluation...')

    for tk in tqdm(top_ks):
        top_k_retrieved_ids = retrieved_ids[:, :tk].copy()
        res_top_k = evaluate_retrieved_results(top_k_retrieved_ids, labels, tk)
        results.append(res_top_k)

    return results


def evaluate_retrieved_results(
    retrieved_ids: np.ndarray,
    labels: np.ndarray,
    top_k: int
) -> Dict[str, float]:

    logger = get_logger()
    logger.info('Computing metrics...')
    targets = (
        np.expand_dims(labels, axis=1) == retrieved_ids
    )
    targets = torch.tensor(np.array(targets), dtype=torch.float16)
    targets = torch.clamp(targets, min=0, max=1)

    preds = torch.tensor(
        np.geomspace(1, 0.1, top_k), dtype=torch.float32
    )
    preds /= torch.sum(preds)
    preds = preds.repeat((targets.shape[0], 1))

    indexes = torch.arange(targets.shape[0]).view(
        -1, 1) * torch.ones(1, targets.shape[1]).long()

    metrics = [
        torchmetrics.retrieval.RetrievalMRR(),
        torchmetrics.retrieval.RetrievalNormalizedDCG(),
        torchmetrics.retrieval.RetrievalPrecision(),
        torchmetrics.retrieval.RetrievalRecall(),
        torchmetrics.retrieval.RetrievalHitRate(),
        torchmetrics.retrieval.RetrievalMAP()
    ]
    results = {}

    for metr in metrics:
        score = round(metr(preds, targets, indexes).item(), 4)
        metr_name = metr.__class__.__name__.replace('Retrieval', '')
        results[metr_name] = score
        logger.info(f'Top-{top_k}: {metr_name}: {score}')

    return results


if __name__ == '__main__':

    initialize(config_path='configs', version_base='1.1')
    config = compose(config_name='config')

    results = run_retrieval_evaluation(config)

    for top_k, res in zip(config.retrieval.evaluation.top_ks, results):
        print(f'{top_k}: {res}')

    # results = run_colbert_retrieval_evaluation(config)
    #
    # for top_k, res in zip(config.retrieval.evaluation.top_ks, results):
    #     print(f'{top_k}: {res}')
