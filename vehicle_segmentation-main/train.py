from model.FCN8s import FCN8s
from Config.config import get_config_dict
from Core.engine_fcn import Trainer





if __name__=='__main__':
    cfg = get_config_dict()
    trainer = Trainer(cfg)
    trainer.training()
