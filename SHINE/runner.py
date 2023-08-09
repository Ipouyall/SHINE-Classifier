from .preprocess.preprocess import make_node2id_eng_text
from .config import Config
from .trainer import Trainer
from .utils import set_seed, save_res


class Runner:
    def __init__(self, config: Config):
        self.config = config

    def run(self):
        set_seed(self.config.seed)

        if self.config.need_preprocess:
            self.__preprocess()

        self.__train()

        save_res(self.config,
                 self.test_acc,
                 self.best_f1)

    def __preprocess(self):
        make_node2id_eng_text(self.config)

    def __train(self):
        trainer = Trainer(self.config)
        self.test_acc, self.best_f1 = trainer.train()
