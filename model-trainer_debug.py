# from accelerate import Accelerator
import copy
from genericpath import exists
import random
from re import L
# from MeCab import Model
from datasets import load_metric, Metric
import json
from loguru import logger
from sqlalchemy import false
from sympy import true
from tap import Tap
import numpy as np
from torch.optim import AdamW
from torch import nn
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
    set_seed,
)
from transformers.models import bart
from typing import Optional, Tuple  # 将wav2vec processor 和 model 合并
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import torch
from typing import Dict, List
import os
from torch.utils.tensorboard import SummaryWriter
import shutil
from utils import EarlyStopping
from processor_zh import DataProcessor, TextDataProcessor, TextInputExample

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



class Config(Tap):
    seed: int = 2022
    pwd: str = '/home/users/jiangjin/jiangjin_bupt/ASR_CORRECTION/Context_Correction/Baseline_and_Other_models/Baseline/Baseline-multi_encoder/'

    output_dir: str =pwd + '/log' 

    device: str = 'cuda'

    text_data_dir: str = pwd + 'data/'
    current_dataset: str = ['AISHELL-1', 'magicdata', 'thchs'][0]

    pretrained_model: str = pwd + 'pretrained-model/BART'

    # Model_config = AutoConfig.from_pretrained(pretrained_model)


    batch_size: int = 4
    shuffle: bool = True
    max_seq_length: int = 120
    learning_rate: float = 5e-5
    weight_decay: float = 0.02
    lr_scheduler_type: str = 'linear'
    num_warmup_steps: int = 20
    max_train_steps: int = 20
    is_debug: bool = False
    gradient_accumulation_steps: int = 1
    epochs: int = 30
    num_batch_per_evaluation: int = 10

    early_stop = EarlyStopping(patience=7)

    metric: str = 'cer'

    early_stop_flag: str = False


    # context-related config：
    model_type: str = 'single-encoder-decoder' # baseline single-encoder-decoder
    if_use_future: bool = False # 是否使用当前句子后面的句子
    context_sents_nub: int = 4 # 使用多少个context句子。单指句子前 or 句子后的 context

    mode_mode_path: str = pwd + model_type 
    best_model_dir: str = mode_mode_path + '/model-checkpoint/'
    test_result_dir: str = mode_mode_path + '/result/'

    mode: str = 'train'

    def get_device(self):
        """return the device"""
        return torch.device(self.device)


class ContextContainer:
    """Context data container for training
    """

    def __init__(self) -> None:
        """init the variables"""
        self.train_step: int = 0
        self.dev_step: int = 0
        self.epoch: int = 0

        self.train_cer: float = 1000
        self.dev_cer: float = 1000
        self.best_dev_cer: float = 1000
        self.test_cer: float = 1000

        self.loss = 0
        self.dev_loss = 0
        self.logits = 0
        self.labels = 0



class Trainer:
    """Trainer which can handle the train/eval/test/predict stage of the model
    """

    def __init__(
        self, config: Config,
        text_processor: DataProcessor,

        text_tokenizer: PreTrainedTokenizer,

        model: PreTrainedModel,
        metric: Metric,
    ) -> None:
        self.config = config

        self.text_tokenizer = text_tokenizer

        model.resize_token_embeddings(len(text_tokenizer))

        # model = nn.DataParallel(model)
        self.model = model.to(self.config.get_device())

        self.metric = metric

        # 2. build text & audio dataloader
        logger.info('init text & audio dataloaders ...')
        self.train_dataloader = self.create_dataloader(
            dataset=text_processor.get_train_dataset(),
            shuffle=self.config.shuffle,
            collate_fn=self.convert_examples_to_features,
        )
        self.dev_dataloader = self.create_dataloader(
            dataset=text_processor.get_dev_dataset(),
            shuffle=False,
            collate_fn=self.convert_examples_to_features,
        )
        self.test_dataloader = self.create_dataloader(
            dataset=text_processor.get_test_dataset(),
            shuffle=False,
            collate_fn=self.convert_examples_to_features,
        )

        # 3. init model related
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=config.learning_rate
                               )

        self.config.max_train_steps = len(self.train_dataloader) * self.config.epochs

        self.lr_scheduler = get_scheduler(
            name=config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=config.num_warmup_steps, # 前 * step 进行warm up（即让lr 从0-设定的lr）
            num_training_steps=config.max_train_steps, # 最大的step
        )

        self.context_data = ContextContainer()
        self._init_output_dir()
       
        self.train_bar: tqdm = None


    def create_dataloader(self, dataset: Dataset, collate_fn, shuffle) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle, # self.config.shuffle,
            collate_fn=collate_fn
        )

    def convert_examples_to_features(self, examples: List[TextInputExample]):
        """convert the examples to features"""
        texts = [example.rec for example in examples]
        encoded_features = self.text_tokenizer.batch_encode_plus(
            texts,
            max_length=self.config.max_seq_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )

        labels = [example.lab for example in examples]
        label_features = self.text_tokenizer.batch_encode_plus(
            labels,
            max_length=self.config.max_seq_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        return encoded_features['input_ids'], label_features['input_ids']

    def _init_output_dir(self):
        logger.info(f'init the output dir: {self.config.output_dir}')
        if os.path.exists(self.config.output_dir):
            pass
            # shutil.rmtree(self.config.output_dir)
        else:
            os.makedirs(self.config.output_dir)

    def _update_train_bar(self):
        infos = [f'epoch: {self.context_data.epoch}/{self.config.epochs}']

        loss = self.context_data.loss
        if torch.is_tensor(loss):
            loss = loss.detach().clone().cpu().numpy().item()
        infos.append(f'loss: <{loss}>')

        self.train_bar.update()
        self.train_bar.set_description('\t'.join(infos))

    def on_batch_start(self):
        '''handle the on batch start logits
        '''
        self.model.train()

    def on_batch_end(self):
        """handle the on batch training is ending logits
        """
        # 1. update global step
        # self.context_data.train_step += 1

        self._update_train_bar()
        ### best model 的model
        # 2. compute cer on training dataset
        # if self.context_data.train_step % config.num_batch_per_evaluation == 0:
        #     self.context_data.train_cer = self.evaluate(self.train_dataloader)
        #     self.writer.add_scalar(
        #         'train/cer',
        #         scalar_value=self.context_data.train_cer,
        #         global_step=self.context_data.train_step,
        #     )



        # self.context_data.dev_step += 1
        # config.num_batch_per_evaluation = len(self.train_dataloader)
        # if self.context_data.dev_step % config.num_batch_per_evaluation == 0:
        #     self.context_data.dev_cer = self.evaluate(self.dev_dataloader)
        #     self.config.early_stop_flag = self.config.early_stop.step(self.context_data.dev_cer)
        #     # dev_cer = self.evaluate(self.dev_dataloader)
        #     # if self.context_data.dev_cer == dev_cer == self.context_data.dev_early_stop_cer:
        #     #     self.context_data.early_stop = True
        #     # self.context_data.dev_early_stop_cer = self.context_data.dev_cer
        #     # self.context_data.dev_cer = dev_cer
        #     self.on_evaluation_end(self.context_data.dev_cer)
        #     # self.writer.add_scalar(
        #     #     tag='dev/cer',
        #     #     scalar_value=self.context_data.dev_cer,
        #     #     global_step=self.context_data.dev_step
        #     # )
        #     logger.info(f'dev/cer is {self.context_data.dev_cer}')

    def train_epoch(self):
        """handle the logit of training epoch

        Args:
            epoch (int): _description_
        # """

        logger.info(f'training epoch<{self.context_data.epoch}> ...')
        self.train_bar = tqdm(total=len(self.train_dataloader))

        
        for text_batch in self.train_dataloader:
            input_ids, labels = text_batch
            input_ids, labels = input_ids.to(
                self.config.get_device()), labels.to(self.config.get_device())

            self.on_batch_start()

            self.optimizer.zero_grad()

            # forward on text data
            output: Seq2SeqLMOutput = self.model(
                input_ids=input_ids, labels=labels)

            self.context_data.loss = output.loss.mean().detach().cpu().numpy().item()

            output.loss.mean().backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            if self.config.early_stop_flag:
                logger.info('early stopping')
                break

            self.on_batch_end()

    def evaluate(self, dataloader,):
        """handle the logit of evaluating

        Args:
            epoch (int): _description_
        """
        self.model.eval()

        all_decoded_preds = []
        all_decoded_labels = []
        # 这里因为tqdm 中包含 tqdm 所以，暂时采用logger方式
        # for text_batch in tqdm(dataloader, desc='evaluation stage ...'):
        for text_batch in dataloader:
            with torch.no_grad():
                input_ids, labels = text_batch
                input_ids, labels = input_ids.to(
                    self.config.get_device()), labels.to(self.config.get_device())

                # forward on dev/test data
                # add .module for multi-GPU
                generated_tokens: Seq2SeqLMOutput = self.model.generate(
                    input_ids=input_ids)

                generated_tokens = generated_tokens.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                decoded_preds = self.text_tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True)
                decoded_labels = self.text_tokenizer.batch_decode(
                    labels, skip_special_tokens=True)

                decoded_preds = [decoded_pred.replace(' ','') for decoded_pred in decoded_preds]
                decoded_labels = [decoded_label.replace(' ','') for decoded_label in decoded_labels]

                all_decoded_preds = all_decoded_preds + decoded_preds
                all_decoded_labels = all_decoded_labels + decoded_labels

        metric_score = self.metric.compute(
            predictions=all_decoded_preds, references=all_decoded_labels)

        self.model.train()
        return metric_score

    def on_evaluation_end(self, metric_score):
        '''always save the best model'''
        if self.context_data.best_dev_cer > metric_score:
            self.save_model(self.config.best_model_dir)
            self.context_data.best_dev_cer = metric_score
            # self.writer.add_scalar(
            #     tag='dev/best_cer',
            #     scalar_value=self.context_data.dev_cer,
            #     global_step=self.context_data.dev_step
            # )
            logger.info('\n')
            logger.info(f'dev/best_cer is {self.context_data.dev_cer}')
            # self.predict()

    def save_model(self, path):
        if os.path.exists(path):
            pass
            # shutil.rmtree(self.config.output_dir)
        else:
            os.makedirs(path)
        torch.save(self.model.state_dict(), path+'/checkpoint_best.pt')

    def train(self):
        """the main train epoch"""
        logger.info('start training ...')
        logger.info(f'  num example = {len(self.train_dataloader)}')
        logger.info(f'  num epochs = {self.config.epochs}')
        logger.info(f'  total train batch size (parallel) = {self.config.batch_size}' )
        logger.info(f'  total optimization step = {self.config.max_train_steps}')

        self.on_train_start()
        for _ in range(self.config.epochs):            
            self.context_data.epoch += 1
            self.train_epoch()

            self.on_epoch_end()
            if self.config.early_stop_flag:
                logger.info('early stopping on train epoch')
                break

    def on_epoch_end(self):
        self.context_data.dev_cer = self.evaluate(self.dev_dataloader)
        self.config.early_stop_flag = self.config.early_stop.step(self.context_data.dev_cer)
        logger.info('\n')
        logger.info(f'dev/cer is {self.context_data.dev_cer}')
        self.on_evaluation_end(self.context_data.dev_cer)
        


    def on_train_start(self):
        '''inite the dev and test cer'''
        # self.context_data.dev_cer = self.evaluate(self.dev_dataloader)
        # self.context_data.test_cer = self.evaluate(self.test_dataloader)
        # self.writer.add_scalar(
        #     tag='dev/cer',
        #     # scalar_value=self.context_data.dev_cer,
        #     scalar_value=0.2701,
        #     global_step=self.context_data.dev_step
        # )
        # self.writer.add_scalar(
        #     tag='test/cer',
        #     # scalar_value=self.context_data.test_cer,
        #     scalar_value=0.2431,
        #     global_step=self.context_data.dev_step
        # )

    def predict(self, model_path: Optional[str] = None,):
        """ predict the example
            test_dataset = ['test_aidatatang', 'test_magicdata', 'test_thchs']
        """
        # self.load_model(self.config.best_model_dir)
        dataloader = self.test_dataloader
        logger.info('start predicting ...')
        # if model_path is not None:
        #     self.load_model(self.config.best_model_dir + model_path)
        self.load_model(self.config.best_model_dir + 'checkpoint_best.pt')

        self.model.eval()

        all_decoded_preds = []
        all_decoded_labels = []

        for text_batch in dataloader:
            with torch.no_grad():
                input_ids, labels = text_batch
                input_ids, labels = input_ids.to(
                    self.config.get_device()), labels.to(self.config.get_device())

                # forward on dev/test data
                # add .module for multi-GPU
                generated_tokens: Seq2SeqLMOutput = self.model.generate(
                    input_ids=input_ids)

                generated_tokens = generated_tokens.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                decoded_preds = self.text_tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True)
                decoded_labels = self.text_tokenizer.batch_decode(
                    labels, skip_special_tokens=True)

                decoded_preds = [decoded_pred.replace(' ','') for decoded_pred in decoded_preds]
                decoded_labels = [decoded_label.replace(' ','') for decoded_label in decoded_labels]

                all_decoded_preds = all_decoded_preds + decoded_preds
                all_decoded_labels = all_decoded_labels + decoded_labels

        metric_score = self.metric.compute(
            predictions=all_decoded_preds, references=all_decoded_labels)

        self.save_test_result(all_decoded_preds, all_decoded_labels, self.config.current_dataset)

        self.context_data.test_cer = metric_score
        # self.writer.add_scalar(
        #     tag='test/'+self.config.current_dataset+'_cer',
        #     scalar_value=self.context_data.test_cer,
        #     global_step=self.context_data.dev_step
        # )
        logger.info(f'test/cer is {self.context_data.test_cer}')
        # add test cer every time evaluate test data
        self.model.train()
        return metric_score

    def load_model(self, path):
        logger.info('load model ...')
        self.model.load_state_dict(torch.load(path))

    def save_test_result(self, all_decoded_preds, all_decoded_labels, test_data_name):
        # for text_modal: add additional 'text_modal_' to distinguish
        # ['cross_modal', 'text_modal']
        if os.path.exists(self.config.test_result_dir):
            pass
        else:
            os.makedirs(self.config.test_result_dir) 
        with open(config.test_result_dir+'T_modal_'+test_data_name+'.txt', 'w') as f_result:
            data_output_list = []
            for item_pred, item_label in zip(all_decoded_preds, all_decoded_labels):
                data_output_list.append(item_pred + ' ' + item_label + '\n') 
            f_result.writelines(data_output_list)


def set_my_seed(seed):
    '''random:
        python
        Numpy'''
    set_seed(seed)
    from torch.backends import cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True



if __name__ == "__main__":
    config: Config = Config().parse_args(known_only=True)

    set_my_seed(config.seed)
    if os.path.exists(config.mode_mode_path):
        pass
            # shutil.rmtree(self.config.output_dir)
    else:
        os.makedirs(config.mode_mode_path)
        

    trainer = Trainer(
        config,
        text_processor=TextDataProcessor(
            config.text_data_dir, config),
        text_tokenizer=AutoTokenizer.from_pretrained(config.pretrained_model),
        model=AutoModelForSeq2SeqLM.from_pretrained(config.pretrained_model),
        # model = AutoModelForSeq2SeqLM.from_config(config.Model_config),
        metric=load_metric(config.metric)
    )
    if config.mode == 'train':
        logger.add(os.path.join(config.mode_mode_path, 'log/train.'+config.current_dataset+'.T-model-log.txt'))
        trainer.train()
    elif config.mode == 'test':
        logger.add(os.path.join(config.mode_mode_path, 'log/test.'+config.current_dataset+'.T-model-log.txt'))
        trainer.predict()



