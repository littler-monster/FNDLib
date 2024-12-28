import time
from conf.config import Config
from util.DataLoader import load_data, load_graph_data, load_mul_data
from FNDLib import FNDLib
import torch
from util.tool import set_seed


if __name__ =='__main__':

    # 1. Load configuration
    cfg = Config()

    # 2. Import Fake news Detection Model
    device = cfg.device
    seed = cfg.seed
    set_seed(seed)
    torch.set_num_threads(16)
    model_str = 'from model.'+cfg.model_type+"."+cfg.model_name+' import '+cfg.model_name
    exec(model_str)

    # 3. Load Data
    mode = "train"
    if cfg.model_type == "textBased":
        data = load_data(cfg)
    elif cfg.model_type == "graphBased":
        data = load_graph_data(cfg)
    elif cfg.model_type == "mulBased":
        data = load_mul_data(cfg, mode)
    elif cfg.model_type == "llmBased":
        pass
    
   

    # data = load_data(cfg)

    # 4.Define Fake news Detection Model and define FNDLib to control the process
    print("start eval")
    Model = eval(cfg.model_name)(cfg)
    print("finish eval")
    # Model = eval(cfg.model_name)(cfg)
    if cfg.model_type == "textBased":
        cls = 0
    elif cfg.model_type == "graphBased":
        cls = 1
    elif cfg.model_type == "mulBased":
        cls = 2
    elif cfg.model_type == "llmBased":
        cls = 3


    dndLib = FNDLib(Model, data, cfg, cls)

    s = time.time()

    # 5. Train and test
    dndLib.Train()
    dndLib.Test()

    # 6. Draw the process of train
    dndLib.DrawLoss()

    e = time.time()

    print("Finish~")
    print("Running time: %f s" % (e - s))