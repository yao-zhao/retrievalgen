import nlp

ELI5 = nlp.load_dataset('eli5')
WIKI40B_SNIPPETS = nlp.load_dataset('wiki_snippets', name='wiki40b_en_100_0')['train']
SAVED_RETRIEVER = "retriever_models/eli5_retriever_model_l-8_h-768_b-512-512"
SAVED_INDEX = "wiki_index/wiki40b_passages_reps_32_l-8_h-768_b-512-512.dat"


class ArgumentsQAR():
    def __init__(self):
        self.batch_size = 512
        self.max_length = 128
        self.checkpoint_batch_size = 32
        self.print_freq = 100
        self.pretrained_model_name = "google/bert_uncased_L-8_H-768_A-12"
        self.model_save_name = SAVED_RETRIEVER
        self.learning_rate = 2e-4
        self.num_epochs = 10
