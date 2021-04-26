import lfqa_utils
import utils

qar_args = utils.ArgumentsQAR()

# prepare torch Dataset objects
qar_train_dataset = lfqa_utils.ELI5DatasetQARetriver(utils.ELI5['train_eli5'], training=True)
qar_validation_dataset = lfqa_utils.ELI5DatasetQARetriver(utils.ELI5['validation_eli5'], training=False)

# load pre-trained BERT and make model
qar_tokenizer, qar_model = lfqa_utils.make_qa_retriever_model(
    model_name=qar_args.pretrained_model_name,
    from_file=None,
    device="cuda:0"

)

# train the model
lfqa_utils.train_qa_retriever(qar_model, qar_tokenizer, qar_train_dataset, qar_validation_dataset, qar_args)
