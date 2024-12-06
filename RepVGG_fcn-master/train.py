
from Config.config import get_config_dict
from Core.engine_repvggfcn import Trainer





if __name__=='__main__':
    cfg = get_config_dict()
    trainer = Trainer(cfg = cfg)
    trainer.training()
