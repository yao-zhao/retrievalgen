import json
from tqdm import tqdm

import faiss
import numpy as np
import pandas as pd

import lfqa_utils
import utils


wiki40b_passage_reps = np.memmap(
    utils.SAVED_REPRESENTATIONS,
    dtype='float32', mode='r',
    shape=(utils.WIKI40B_SNIPPETS.num_rows, 128)
)

wiki40b_index_flat = faiss.IndexFlatIP(128)
wiki40b_gpu_index = faiss.index_cpu_to_all_gpus(wiki40b_index_flat)
wiki40b_gpu_index.add(wiki40b_passage_reps)

qar_args = utils.ArgumentsQAR()
qar_tokenizer, qar_model = lfqa_utils.make_qa_retriever_model(
    model_name=qar_args.pretrained_model_name,
    from_file=f"{utils.SAVED_RETRIEVER}_9.pth",
    device="cuda:0"
)
_ = qar_model.eval()

batch_size = 32

for split in ["validation_eli5", "test_eli5", "train_eli5"]:
  print(f"processing {split}")
  with open(f"output/{split}.jsonl", "w") as f:
    datasets = utils.ELI5[split]
    for i in tqdm(range(0, len(datasets), batch_size)):
      rows = [datasets[j] for j in range(i, min(i+batch_size, len(datasets)))]
      questions = [r["title"] for r in rows]
      _, all_res_lists = lfqa_utils.batch_query_qa_dense_index(questions, qar_model, qar_tokenizer, utils.WIKI40B_SNIPPETS,
                                                wiki40b_gpu_index)
      for row, res_list in zip(rows, all_res_lists):
        row["doc"] = res_list
        f.write(json.dumps(row) + "\n")
