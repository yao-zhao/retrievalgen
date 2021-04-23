from lfqa_utils import *
from utils import ArgumentsQAR, ELI5

qar_args = ArgumentsQAR()

# prepare torch Dataset objects
qar_train_dataset = ELI5DatasetQARetriver(ELI5['train_eli5'], training=True)
qar_validation_dataset = ELI5DatasetQARetriver(ELI5['validation_eli5'], training=False)

# load pre-trained BERT and make model
qar_tokenizer, qar_model = make_qa_retriever_model(
    model_name=qar_args.pretrained_model_name,
    from_file=None,
    device="cuda:0"

)

# train the model
train_qa_retriever(qar_model, qar_tokenizer, qar_train_dataset, qar_validation_dataset, qar_args)
