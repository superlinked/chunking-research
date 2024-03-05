# chunking-research

This project is a framework to run experiments related to Chunking and Retrieval of a [Retrieval
Augmented Generation (RAG)](https://arxiv.org/abs/2312.10997) system. The framework supports Chunking Methods from [LlamaIndex](https://docs.llamaindex.ai/en/stable/index.html)
and [LangChain](https://www.langchain.com/), Embedding models (single-vector) and Rerankers from Huggingface [Massive Text Embedding Benchmark (MTEB)](https://huggingface.co/spaces/mteb/leaderboard)
leaderboard, and multi-vector Embedding Model [ColBERT](https://huggingface.co/colbert-ir/colbertv2.0) through
[RAGatouille](https://github.com/bclavie/RAGatouille/tree/main). The retrieved documents are evaluated with metrics of
MRR, NDCG@k, Recall@k, Precision@k, MAP@k and Hit-Rate provided by [Torchmetrics](https://lightning.ai/docs/torchmetrics/stable/).
The combination of the Chunking Methods, Embedding Models and Rerankers are evaluated on Information Retrieval (IR) datasets, specifically,
Question-Answering (QA) datasets. These datasets contain `<question, context, answer>` triples. Therefore, they are suitable for the evaluation of
end to end RAG systems. For the evaluation of the Retrieval component, the `<question, context>` pairs are sufficient.

The following datasets are supported currently:

- [SQUAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/)
- [QuAC](https://quac.ai/)
- [HotpotQA](https://hotpotqa.github.io/)

These datasets are also available through [Huggingface Question Answering Datasets](https://huggingface.co/datasets?task_categories=task_categories:question-answering&sort=trending).
The framework is easily extendable with additional QA datasets by following the output conventions of the preprocessors.

For document chunking, the following methods are supported:

- [SentenceSplitter](https://docs.llamaindex.ai/en/stable/api/llama_index.core.node_parser.SentenceSplitter.html#llama_index.core.node_parser.SentenceSplitter)
- [SemanticSplitterNodeParser](https://docs.llamaindex.ai/en/stable/api/llama_index.core.node_parser.SemanticSplitterNodeParser.html#llama_index.core.node_parser.SemanticSplitterNodeParser)
- [TokenTextSplitter](https://docs.llamaindex.ai/en/stable/api/llama_index.core.node_parser.TokenTextSplitter.html#llama_index.core.node_parser.TokenTextSplitter)
- [SentenceWindowNodeParser](https://docs.llamaindex.ai/en/stable/api/llama_index.core.node_parser.SentenceWindowNodeParser.html#llama_index.core.node_parser.SentenceWindowNodeParser)
- [LangchainNodeParser](https://docs.llamaindex.ai/en/stable/api/llama_index.core.node_parser.LangchainNodeParser.html#llama_index.core.node_parser.LangchainNodeParser)

The LangchainNodeParser supports [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter)
from LangChain. Examples can be found [here](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules.html).

## Configuration

The experiments are configuration driven with [Hydra](https://hydra.cc/docs/intro/). The main settings in the 
configuration file are the following:

**config.yaml**

Experiments can be run based on either the "config.yaml" or the "pipeline.yaml" file. If the pipeline is empty or 
everything is commented out, the main config will be used. There are two types of experiments, Huggingface Embedding
Model or ColBERT based experiments. The ```mlflow.experiment_name``` section defines which experiment type will be run.
Its values can be ```Retrieval``` or ```RetrievalColBERT``` that define the underlying ```ExperimentRunner``` classes.

```
defaults:
  - _self_
  - pipeline: pipeline

mlflow:
  experiment_name: Retrieval
  run_name: BAAI/bge-small-en-v1.5
  description: test run
  tags:
    experiment: Retrieval
    dataset: QUAC
    model: BAAI/bge-small-en-v1.5
```

Paths to the datasets, saved question embeddings, MLflow experiments and the ColBERT index. Currently, only the question embeddings
are saved, because these are independent of the chunking method and its parameters. The context embeddings could be saved and re-used
as well by creating directories from hashing the name of the chunking method and its parameters. ColBERT default saving path is overwritten
by the path below, but it is not used for re-loading, the index is overwritten by new ColBERT experiments.

```
datasets: ./datasets/
embeddings: ./artifacts/embeddings/
experiments: ./artifacts/experiments/
colbert_index: ./artifacts/colbert_index
```

There are 4 main sections related to Huggingface Embedding Models based retrieval and ranking. 

The chunker
section defines the chunking method to be used and its parameters. **It is important to note that everything below the chunker until the model**
section should be commented out if the pipeline mode is used to run the experiments, for proper configuration file saving to MLflow.

The model section defines the Huggingface Embedding Model and its parameters. The normalize parameter should be set to true.

Reranking after Retrieval is turned on with the ```reranking``` flag. If it is turned on, the ```reranker``` section defines the model to be used
and its parameters. It is important to note that on Windows OS, using multiple workers for data loading resulted in bugs and wrong outputs.

The evaluation section defines the top k-s for which the Retrieval metrics will be computed.

```
retrieval:
  # comment everything below the chunker until the model if the pipeline is used
  chunker:
    name: SentenceSplitter
    params:
      chunk_size: 128
      chunk_overlap: 16
      
  model:
   model_name: BAAI/bge-small-en-v1.5
   embed_batch_size: 32
   normalize: true
  
  reranking: false
  reranker:
    model_name: cross-encoder/ms-marco-TinyBERT-L-2-v2
    predict:
      batch_size: 32
      show_progress_bar: true
      #num_workers: 6 DON'T use it, it is buggy, same scores are returned!
      
  evaluation:
    top_ks:
      - 1
      - 3
      - 5
      - 7
      - 10
```

For ColBERT based Retrieval the following section defines the supported parameters. By default,
RAGatouille and ColBERT supports [LlamaIndex SentenceSplitter](https://docs.llamaindex.ai/en/stable/api/llama_index.core.node_parser.SentenceSplitter.html#llama_index.core.node_parser.SentenceSplitter)
chunking method. The ```max_document_length```  parameter defines the maximum chunk size. The ```question_batch_size```
parameter defines the batch size to be applied for Retrieval evaluation.

```
colbert_retrieval:
  model_name: colbert-ir/colbertv2.0
  max_document_length: 128
  question_batch_size: 5000
  evaluation:
    top_ks:
      - 1
      - 3
      - 5
      - 7
      - 10
```

## Installation

Installation is complicated due to RAGatouille, ColBERT and Faiss-GPU (the latter is a must for optimized runtime).

The following steps were tested on Ubuntu 22.04.2 LTS. To install the proper Nvidia driver and CUDA version the easiest
way, follow the steps listed [in this blog](https://www.cherryservers.com/blog/install-cuda-ubuntu). For convenience, the
steps are listed and extended below.

1. sudo ubuntu-drivers devices
2. sudo apt install nvidia-driver-470
3. sudo reboot now
4. nvidia-smi
5. sudo apt install gcc
6. gcc -v
7. [Download and install CUDA toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) 
from runfile (local) with these options: --toolkit --silent --override
8. sudo reboot
9. nano ~/.bashrc
10. export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
11. export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
12. ~/.bashrc
13. nvcc -V

`conda create -n chunking-research python=3.10.13`

`conda activate chunking-research`

`pip install -r requirements.txt`

Unfortunately, RAGatouille's LlamaIndex dependency is inconsistent, so LlamaIndex should be removed and re-installed.

`pip uninstall llama-index`

`pip install llama-index --upgrade --no-cache-dir --force-reinstall`

Faiss-CPU should be also removed and Faiss-GPU to be **installed with conda**, as described [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#installing-from-conda-forge).

`pip uninstall faiss-cpu`

`conda install -c conda-forge faiss-gpu`

## Start mlflow server

`mlflow server --backend-store-uri artifacts/experiments/`

## Run the experiments based on the config

`python experiments.py`

or without saving the results to MLflow

`python retrieval.py`

## Share / load pre-computed artifacts

Download the experiment results - "experiments/" from
[this Google Drive link](https://drive.google.com/drive/folders/1LvgXt8sriRBEwrgOZvcyIyJUKjPMz51Q?ths=true),
and place them under the "./artifacts/" folder, then run the mlflow server with the command above.
