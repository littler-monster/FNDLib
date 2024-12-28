import argparse


def Config():
    parser = argparse.ArgumentParser()

    # dataset #
    parser.add_argument("--news_npy_path", type=str, default=".//data//Politifact//news_politic_2.npy")
    parser.add_argument("--comment_npy_path", type=str, default=".//data//Politifact//comments_politic_2.npy")
    parser.add_argument("--label_npy_path", type=str, default=".//data//Politifact//label_politic_2.npy")

    parser.add_argument("--news_npy_path_g", type=str, default="/usr/gao/gubincheng/article_rep/Data/GALA_data/Classes/gossip_0/news_gossip_0.npy")
    parser.add_argument("--comment_npy_path_g", type=str, default="/usr/gao/gubincheng/article_rep/Data/GALA_data/Classes/gossip_0/comments_gossip_0.npy")
    parser.add_argument("--label_npy_path_g", type=str, default="/usr/gao/gubincheng/article_rep/Data/GALA_data/Classes/gossip_0/label_gossip_0.npy")

    parser.add_argument("--news_npy_path_p_7", type=str, default="/usr/gao/gubincheng/article_rep/Data/GALA_data/Classes/politic_7/news_politic_7.npy")
    parser.add_argument("--comment_npy_path_p_7", type=str, default="/usr/gao/gubincheng/article_rep/Data/GALA_data/Classes/politic_7/comments_politic_7.npy")
    parser.add_argument("--label_npy_path_p_7", type=str, default="/usr/gao/gubincheng/article_rep/Data/GALA_data/Classes/politic_7/label_politic_7.npy")   

    parser.add_argument("--graph_label_path", type=str, default="data/Twitter15/Twitter15_label_All.txt")  # BiGCN:Twitter15  GACL Twitter16
    parser.add_argument("--graph_path", type=str, default=".//data//Twitter15graph")
    parser.add_argument("--data_path_GACL", type=str, default="./data/Twitter16/twitter16")
    parser.add_argument("--datasetname", type=str, default="Twitter15")   # BiGCN:Twitter15  GACL Twitter16

    # mul data
    parser.add_argument("--mul_politic_body", type=str, default="/usr/gao/gubincheng/article_rep/MBO_data/politic/glove/politic_body.npy")
    parser.add_argument("--mul_politic_img", type=str, default="/usr/gao/gubincheng/article_rep/MBO_data/politic/glove/politic_img.npy")
    parser.add_argument("--mul_politic_label", type=str, default="/usr/gao/gubincheng/article_rep/MBO_data/politic/glove/politic_label.npy")

    parser.add_argument("--mul_gossip_body", type=str, default="/usr/gao/gubincheng/article_rep/MBO_data/gossip/glove/gossip_body.npy")
    parser.add_argument("--mul_gossip_img", type=str, default="/usr/gao/gubincheng/article_rep/MBO_data/gossip/glove/gossip_img.npy")
    parser.add_argument("--mul_gossip_label", type=str, default="/usr/gao/gubincheng/article_rep/MBO_data/gossip/glove/gossip_label.npy")


    # model #
    parser.add_argument("--model_name", type=str, default="dEFEND") # 选择方法名
    parser.add_argument("--model_type", type=str, default="textBased",help="[graphBased,llmBased,textBased,mulBased]")
    parser.add_argument("--comments_need",type=bool, default=True, help="need comments or not")
    parser.add_argument("--num_class", type=int, default="2")
    parser.add_argument("--device", type=str, default="cuda:0", help="[cuda, cuda:0, cuda:1, cpu]")
    parser.add_argument("--seed", type=int, default="42", help="random seed")
    parser.add_argument("--weight_decay", type=float, default="0.001", help="optimizer punish decay ratio")
    parser.add_argument("--lr", type=float, default="0.001", help="learing rate")
    parser.add_argument("--maxEpoch", type=int, default="100", help="number of epochs")
    parser.add_argument("--batch_size", type=int, default="64", help="batch size")
    parser.add_argument("--patience", type=int, default="8", help="early_stop epoch")
    parser.add_argument("--percent_of_val", type=float, default="0.2", help="val size")
    parser.add_argument("--percent_of_test", type=float, default="0.2", help="test size")
    parser.add_argument("--alpha_HMCAN", type=float, default="0.5", help="alpha")
    # 按需修改
    parser.add_argument("--graph_in_feats", type=int, default=5000)  #GACL:768 BiGCN:5000
    parser.add_argument("--graph_out_feats", type=int, default=64)   
    parser.add_argument("--graph_hid_feats", type=int, default=64)   

    parser.add_argument("--TDdroprate", type=float, default=0.2)
    parser.add_argument("--BUdroprate", type=float, default=0.2)
    parser.add_argument("--GACLdroprate", type=float, default=0.4)
    
    # HMCAN
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max_word_length", type=int, default=8)
    parser.add_argument("--contextual_transform_output_dim", type=int, default=100)
    parser.add_argument("--contextual_transform_input_fc", type=bool, default=False)
    parser.add_argument("--contextual_transform_num_layers", type=int, default=2)
    parser.add_argument("--contextual_transform_num_heads", type=int, default=10)
    parser.add_argument("--contextual_transform_dropout", type=float, default=0.1)
    parser.add_argument("--contextual_transform_use_context", type=bool, default=True)
    parser.add_argument("--contextual_transform_atn_ct_num_layers", type=int, default=2)
    parser.add_argument("--contextual_transform_atn_ct_num_heads", type=int, default=2)
    parser.add_argument("--contextual_transform_pooler", type=str, default="avg")

    
    # save model#
    parser.add_argument("--save", type=bool, default=True, help="save model or not")
    parser.add_argument("--save_dir", type=str, default="./modelsaved/dEFEND", help="output directory for model")

    # glove or word2vec
    parser.add_argument("--glove_path", type=str, default="your path",help="the path of glove")
    parser.add_argument("--glove_txt_path", type=str, default="your path", help="the path of dictionary")
    parser.add_argument("--bert_w2c_path", type=str, default="/usr/gao/gubincheng/article_rep/FNDlib/FNDLib_v.1.0.0/data/bert_w2c", help="bert_pre")

    return parser.parse_args()
