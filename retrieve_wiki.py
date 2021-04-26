import faiss
import numpy as np
import pandas as pd

import lfqa_utils
import utils

# faiss_res = faiss.StandardGpuResources()
wiki40b_passage_reps = np.memmap(
    utils.SAVED_REPRESENTATIONS,
    dtype='float32', mode='r',
    shape=(utils.WIKI40B_SNIPPETS.num_rows, 128)
)

wiki40b_index_flat = faiss.IndexFlatIP(128)
wiki40b_gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
    wiki40b_index_flat
)
# wiki40b_gpu_index = faiss.index_cpu_to_gpu_multiple(faiss_res, [1], wiki40b_index_flat)
wiki40b_gpu_index.add(wiki40b_passage_reps)

qar_args = utils.ArgumentsQAR()
qar_tokenizer, qar_model = lfqa_utils.make_qa_retriever_model(
    model_name=qar_args.pretrained_model_name,
    from_file=f"{utils.SAVED_RETRIEVER}_9.pth",
    device="cuda:1"
)
_ = qar_model.eval()

question = utils.ELI5['test_eli5'][12345]['title']
doc, res_list = lfqa_utils.query_qa_dense_index(question, qar_model, qar_tokenizer, utils.WIKI40B_SNIPPETS,
                                                wiki40b_gpu_index, device='cuda:1')

df = pd.DataFrame({
    'Article': ['---'] + [res['article_title'] for res in res_list],
    'Sections': ['---'] + [res['section_title'] if res['section_title'].strip() != '' else res['article_title']
                           for res in res_list],
    'Text': ['--- ' + question] + [res['passage_text'] for res in res_list],
})
df.style.set_properties(**{'text-align': 'left'})
print(df)
