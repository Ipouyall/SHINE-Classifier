import os.path
from dataclasses import dataclass
from typing import Iterable, Union, Tuple

import torch


@dataclass()
class Config:
    delete_stopwords: bool = False
    stopwords_path: Union[str, None] = "./SHONE/preprocess/stopwords_en.txt"
    raw_data_path: Union[str, None] = None  # if preprocess needed, place the original data path
    force_preprocess: bool = False

    data_path: str = "./data"
    dataset: str = "olidv2"  # by default, 'snippets', 'twitter', and 'olidv2' are available


    save_path: str = "./"
    disable_cuda: bool = False

    seed: int = 119

    hidden_size: int = 200

    threshold: float = 2.7
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    drop_out: float = 0.7
    max_epoch: int = 500

    concat_word_emb: bool = True

    type_num_node: Tuple[str] = ('query', 'tag', 'word', 'entity')

    def __post_init__(self):
        self.device = 'cuda' if torch.cuda.is_available() and (not self.disable_cuda) else 'cpu'
        self.data_path = self.data_path + f'/{self.dataset}_data/'
        self.save_name = self.save_path + f'./result_{self.dataset}.json'
        self.need_preprocess = self.force_preprocess

        if not os.path.exists(self.need_preprocess):
            print(f"Can't find specified dataset at {self.data_path}")
            raise FileNotFoundError()

        if not os.path.exists(self.data_path) and not self.need_preprocess:
            if self.raw_data_path is not None:
                self.need_preprocess = True
            else:
                print(f"Couldn't find specified dataset at {self.data_path}")
                raise FileNotFoundError()

        if self.delete_stopwords and self.stopwords_path in None:
            print(f"Couldn't find stopwords file at {self.stopwords_path}\n" +
                  "::We would use NLTK's english stopwords instead")
            self.stopwords_path = None



# set_seed(params.seed)
# trainer = Trainer(params)
# test_acc, best_f1 = trainer.train()
# save_res(params, test_acc, best_f1)