from sentence_transformers import CrossEncoder

from financerag.rerank import CrossEncoderReranker
from financerag.retrieval import DenseRetriever, SentenceTransformersEncoder
from financerag.tasks import FinDER

finder_task = FinDER()

retrieval_model = DenseRetriever(
    model=SentenceTransformersEncoder(model_name_or_path='intfloat/e5-large-v2')
)

retrieval_result = finder_task.retrieve(
    retriever=retrieval_model
)

reranker = CrossEncoderReranker(
    model=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
)

reranking_result = finder_task.rerank(
    reranker=reranker,
    results=retrieval_result,
    top_k=100,
    batch_size=32
)

finder_task.save_results(output_dir='./results')