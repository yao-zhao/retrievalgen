from lfqa_utils import *
from utils import ArgumentsQAR, WIKI40B_SNIPPETS, SAVED_RETRIEVER, SAVED_INDEX

# load pre-trained BERT and make model
qar_args = ArgumentsQAR()
qar_tokenizer, qar_model = make_qa_retriever_model(
    model_name=qar_args.pretrained_model_name,
    from_file=f"{SAVED_RETRIEVER}_9.pth",
    device="cuda:0"
)
_ = qar_model.eval()

if not os.path.isfile(SAVED_INDEX):
    make_qa_dense_index(
        qar_model, qar_tokenizer, WIKI40B_SNIPPETS, device='cuda:0',
        index_name=SAVED_INDEX
    )
