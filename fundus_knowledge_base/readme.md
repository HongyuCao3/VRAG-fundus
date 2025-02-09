# Fundus Knowledge Base

## overview
This module serve as the retriever of VRAG framework composed of three parts: 1) emb_builder; 2) index_manger; 3) knowledge_retriever;

## emb_builder
The emb_builder can convert the image or text to their embeddings and save into folders.
Currently we use index_manager as image embedding, so this part now is only for text embedding generation and saving.

## index_manager
The index_manager can convert the image to their embeddings and use [llama-index](https://github.com/run-llama/llama_index) to auto buid retrievers.

## knowledge_retriever
The knowledge_retriever can retrieve similar texts from image, which is **not** implemented by [llama-index](https://github.com/run-llama/llama_index)
Since the retrieval of images is the function of index_manager, this part is only for text retrieval