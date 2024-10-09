import argparse


def Config():
    parser = argparse.ArgumentParser()

    # dataset #
    parser.add_argument("--dataset_name", type=str, default="Politifact")
    parser.add_argument("--news_npy_path", type=str, default=".//data//Politifact//news_politic_2.npy")
    parser.add_argument("--comment_npy_path", type=str, default=".//data//Politifact//comments_politic_2.npy")
    parser.add_argument("--label_npy_path", type=str, default=".//data//Politifact//label_politic_2.npy")

    # model #
    parser.add_argument("--model_name", type=str, default="dEFEND")
    parser.add_argument("--model_type", type=str, default="textBased",help="[graphBased,llmBased,textBased]")
    parser.add_argument("--comments_need",type=bool, default=True, help="need comments or not")
    parser.add_argument("--num_class", type=int, default="2")
    parser.add_argument("--device", type=str, default="cuda", help="[cuda,cpu]")
    parser.add_argument("--seed", type=int, default="42", help="random seed")
    parser.add_argument("--weight_decay", type=float, default="0.005", help="optimizer punish decay ratio")
    parser.add_argument("--lr", type=float, default="0.05", help="learing rate")
    parser.add_argument("--maxEpoch", type=int, default="100", help="number of epochs")
    parser.add_argument("--batch_size", type=int, default="64", help="batch size")
    parser.add_argument("--patience", type=int, default="8", help="early_stop epoch")
    parser.add_argument("--percent_of_val", type=float, default="0.2", help="val size")
    parser.add_argument("--percent_of_test", type=float, default="0.2", help="test size")

    # save model#
    parser.add_argument("--save", type=bool, default=True, help="save model or not")
    parser.add_argument("--save_dir", type=str, default="./modelsaved/dEFEND", help="output directory for model")

    # glove or word2vec
    parser.add_argument("--glove_path", type=str, default="your path",help="the path of glove")
    parser.add_argument("--glove_txt_path", type=str, default="your path", help="the path of dictionary")

    return parser.parse_args()
