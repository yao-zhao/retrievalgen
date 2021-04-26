import os

import lfqa_utils
import utils

qar_args = utils.ArgumentsQAR()
qar_tokenizer, qar_model = lfqa_utils.make_qa_retriever_model(
    model_name=qar_args.pretrained_model_name,
    from_file=f"{utils.SAVED_RETRIEVER}_9.pth",
    device="cuda:0"
)
_ = qar_model.eval()

if not os.path.isfile(utils.SAVED_REPRESENTATIONS):
    lfqa_utils.make_qa_dense_index(
        qar_model, qar_tokenizer, utils.WIKI40B_SNIPPETS, device='cuda:0',
        index_name=utils.SAVED_REPRESENTATIONS
    )
