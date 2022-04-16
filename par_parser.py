import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="BERT-based Multitask Learning")

    parser.add_argument("--task",
                        default="sentivent",
                        type=str,
                        help='name of task to train')

    parser.add_argument("--train_batch_size",
                        type=int,
                        default=8,
                        help="Batch size for train")
    parser.add_argument("--eval_batch_size",
                        type=int,
                        default=16,
                        help="Batch size for evaluation")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=2e-5,
                        help="Gradient descent learning rate for adam.")

    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Number of gradient descent iterations.")

    parser.add_argument("--data_dir",
                        default='D:/2021-2_NLP_Relation\RL_Relation\sentivent_event_sentence_classification\data\processed',
                        #default='./data',
                        type=str,
                        help="input data dir. data files for the training task")

    parser.add_argument("--model_dir",
                        default='./model',
                        type=str,
                        help="path to model")

    parser.add_argument("--eval_dir",
                        default='./eval',
                        type=str,
                        help='eval dir for evaluation.')

    parser.add_argument("--train_file", default="train.tsv", type=str)
    parser.add_argument("--test_file", default="test.tsv", type=str)
    parser.add_argument("--label_file", default="type_classes_multilabelbinarizer.json", type=str)

    parser.add_argument("--model_name",
                        default='bert-base-uncased',
                        type=str,
                        help='model name or path')

    parser.add_argument("--model_size",
                        default='base',
                        type=str,
                        help='which size of model to use')

    parser.add_argument("--truncate",
                        default=512,
                        type=float,
                        help="Truncate the sequence length to")

    parser.add_argument("--seed",
                        default=19971212,
                        type=int,
                        help='random seed for init')

    parser.add_argument("--loss_weights",
                        nargs='+',
                        type=float,
                        default=[1,1,1,1],
                        help="weight decay if we use it.")

    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="epsilon for adam optimizer")
    parser.add_argument("--scheduler",
                        action='store_true',
                        help="use scheduler to optimizer")
    parser.add_argument("--clip",
                        action='store_true',
                        help="use clip to gradients")

    parser.add_argument("--max_steps",
                        default = -1,
                        type = int)
    parser.add_argument("--save_steps",#patience
                        default=250,
                        type=int,
                        help='checkpoint for update')
    parser.add_argument("--logging_steps",
                        default=250,
                        type=int)
    #transformer
    parser.add_argument("--attention_dropout_rate",
                        default=0.1,
                        type=float,
                        help="transformer attention dropout for fully-connected layers.")
    parser.add_argument("--hidden_dropout_rate",
                        default=0.1,
                        type=float,
                        help="transformer hidden dropout for fully-connected layers.")
    #LSTM
    parser.add_argument("--dropout_rate",
                        default=0.1,
                        type=float,
                        help="dropout for fully-connected layers.")
    parser.add_argument("--num_layers",
                        default=1,
                        type=int,
                        help="num of layers of LSTM.")
    parser.add_argument("--hidden_size",
                        type=int,
                        default=300,
                        help="Number of neurons(vectors) by hidden layer of LSTM. Default is 300.")
    parser.add_argument("--hidden_combined_method",
                        type=str,
                        default='concat',
                        help="how to combine hidden vectors in LSTM.")


    parser.add_argument("--test_size",
                        type=float,
                        default=0.20,
                        help="Size of test dataset. Default is 10%.")

    parser.add_argument("--input_size",
                        type=float,
                        default=384,#29496,
                        help="Maximum sequence length after tokenization")

    parser.add_argument("--max_seq_len",
                        type=float,
                        default=50,
                        help="Maximum total input sequence length after tokenization")

    parser.add_argument("--do_train",
                        action="store_true",
                        help="use when running train")
    parser.add_argument("--do_eval",
                        action="store_true",
                        help="use when running eval")
    parser.add_argument("--add_sep_token",
                        action="store_true",
                        help="[SEP] token at the end of sentence"
                        )

    parser.add_argument("--save_model",
                        type=bool,
                        default=True)
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=60733)
    args = parser.parse_args()
    return args