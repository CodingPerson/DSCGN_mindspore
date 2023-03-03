from collections import defaultdict, Counter
import time
import math

import mindspore
from mindspore import load_checkpoint,load_param_into_net
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from math import ceil
import torch
from mindspore import nn, ops
import torch.multiprocessing as mp
from mindspore.ops import operations as P
# import torch.distributed as dist
from mindspore.communication import init, get_group_size, get_rank
from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from mindspore.dataset import DistributedSampler, SequentialSampler, RandomSampler
from transformers import BertTokenizer
from nltk.corpus import stopwords
# from mindformers import  BertTokenizer

import numpy as np
import os
import shutil
import sys
from tqdm import tqdm
from model import LOTClassModel
import warnings
import torch.functional as F
import mindspore.dataset as ds
from loss import GCELoss
from mindspore import Tensor



class TrainDataset():
    def __init__(self,input_ids,attention_masks,labels=None,weights=None):
        self.input_ids =  input_ids
        self.attention_maks = attention_masks
        self.labels = labels
        self.weights = weights
    def __getitem__(self, index):
        index = int(index)
        if self.labels == None and self.weights == None:
            return self.input_ids[index],self.attention_maks[index]
        if self.labels != None and self.weights == None:
            print(self.attention_maks[index])
            return self.input_ids[index], self.attention_maks[index],self.labels[index]
        if self.labels == None and self.weights != None:
            return self.input_ids[index], self.attention_maks[index],self.weights[index]
        if self.labels != None and self.weights != None:
            return self.input_ids[index], self.attention_maks[index], self.labels[index],self.weights[index]
    def __len__(self):
        return 22



class LOTClassTrainer(object):

    def __init__(self, args):
        self.args = args
        self.max_len = args.max_len
        self.dataset_dir = args.dataset_dir
        self.dist_port = args.dist_port
        self.num_cpus = min(10, cpu_count() - 1) if cpu_count() > 1 else 1
        self.world_size = args.gpus
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.accum_steps = args.accum_steps
        eff_batch_size = self.train_batch_size * self.world_size * self.accum_steps
        #assert abs(eff_batch_size - 128) < 10, f"Make sure the effective training batch size is around 128, current: {eff_batch_size}"
        print(f"Effective training batch size: {eff_batch_size}")
        self.pretrained_lm = 'bert-base-uncased'
        #self.pretrained_lm = "/home/ubuntu/chenhu_project/chenhu/LOTClass-master/src/roberta-base"
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_lm)
        #self.tokenizer = RobertaTokenizer.from_pretrained(self.pretrained_lm, do_lower_case=True)
        self.vocab = self.tokenizer.vocab
        self.vocab_size = len(self.vocab)
        self.mask_id = self.vocab[self.tokenizer.mask_token]
        self.inv_vocab = {k:v for v, k in self.vocab.items()}
        self.read_label_names(args.dataset_dir, args.label_names_file)
        self.num_class = len(self.label_name_dict)
        self.model = LOTClassModel(self.num_class)
        self.read_data(args.dataset_dir, args.train_file, args.test_file, args.test_label_file,args.web_file,args.web_label_file)
        self.with_test_label = True if args.test_label_file is not None else False
        self.temp_dir = f'tmp_{self.dist_port}'
        self.mcp_loss = nn.CrossEntropyLoss(reduction='mean')
        self.GCE_loss = GCELoss(q=0.4)
        #self.st_loss = nn.transformer.KLDivLoss(reduction='batchmean')
        self.update_interval = args.update_interval
        self.early_stop = args.early_stop
    def set_up_dist(self, rank):
            # dist.init_process_group(
            #     backend='nccl',
            #     init_method=f'tcp://localhost:{self.dist_port}',
            #     world_size=self.world_size,
            #     rank=rank
            # )
            # init()
            # device_num = get_group_size()
            # rank = get_rank()
            # mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.DATA_PARALLEL)
            # create local model
            model = self.model
            #model = mindspore.Model(model)
            return model

    def get_dict_key(self, value):
        dic = self.vocab
        keys = list(dic.keys())
        values = list(dic.values())
        idx = values.index(value)
        key = keys[idx]

        return key
    # get document truncation statistics with the defined max length
    # def corpus_trunc_stats(self, docs):
    #     doc_len = []
    #     for doc in docs:
    #         input_ids = self.tokenizer.encode(doc, add_special_tokens=True)
    #         doc_len.append(len(input_ids))
    #     print(f"Document max length: {np.max(doc_len)}, avg length: {np.mean(doc_len)}, std length: {np.std(doc_len)}")
    #     trunc_frac = np.sum(np.array(doc_len) > self.max_len) / len(doc_len)
    #     print(f"Truncated fraction of all documents: {trunc_frac}")

    # convert a list of strings to token ids
    def encode(self, docs):
        encoded_dict = self.tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=self.max_len, padding='max_length',
                                                        return_attention_mask=True, truncation=True)

        input_ids = mindspore.Tensor(encoded_dict['input_ids'],dtype=mindspore.dtype.int32)
        attention_masks = mindspore.Tensor(encoded_dict['attention_mask'],dtype=mindspore.dtype.int32)
        return input_ids, attention_masks

    # convert list of token ids to list of strings

    # convert dataset into tensors
    def create_dataset(self, dataset_dir, text_file, label_file, loader_name, find_label_name=False, label_name_loader_name=None):
        loader_file = os.path.join(dataset_dir, loader_name)
        ##先判断是否已经构造完毕J
        if os.path.exists(loader_file):
            print(f"Loading encoded texts from {loader_file}")
            data = torch.load(loader_file)
        else:
            print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
            corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
            docs = [doc.strip() for doc in corpus.readlines()]
            docs = docs[0:int(len(docs))]
            print(f"Converting texts into tensors.")
            chunk_size = ceil(len(docs) / self.num_cpus)
            chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
            results = [self.encode(docs=[doc]) for doc in docs]

            input_ids = []
            attention_masks=[]
            for result in results:
                input_ids.append(result[0])
                attention_masks.append(result[1])

            # inputs = ops.concat(input_ids,axis=0)
            # masks = ops.concat(attention_masks,axis=0)
            #input_ids = ops.concat(input_ids,axis=0)
            #attention_masks = ops.concat(attention_masks,axis=0)
            # input_ids = ops.concat([result[0] for result in results],axis=0)
            # attention_masks = ops.concat([result[1] for result in results],axis=0)

            print(f"Saving encoded texts into {loader_file}")
            if label_file is not None:
                print(f"Reading labels from {os.path.join(dataset_dir, label_file)}")
                truth = open(os.path.join(dataset_dir, label_file))
                ##chenhu
                labels = [int(label.strip().split()[0]) for label in truth.readlines()]
                labels = mindspore.Tensor(labels,dtype=mindspore.dtype.int32)
                data = {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}
            else:
                data = {"input_ids": input_ids, "attention_masks": attention_masks}
        if find_label_name:
            loader_file = os.path.join(dataset_dir, label_name_loader_name)
            if os.path.exists(loader_file):
                print(f"Loading texts with label names from {loader_file}")
                label_name_data = torch.load(loader_file)
            # else:
            #     print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
            #     corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
            #     docs = [doc.strip() for doc in corpus.readlines()]
            #     docs = docs[0:int(len(docs)/50)]
            #     print(len(docs))
            #     print("Locating label names in the corpus.")
            #     chunk_size = ceil(len(docs) / 12)
            #     chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
            #     results = [self.label_name_occurrence(docs=chunk) for chunk in chunks]
            #     #results = Parallel(n_jobs=int(self.num_cpus/2))(delayed(self.label_name_occurrence)(docs=chunk) for chunk in chunks)
            #     input_ids_with_label_name = ops.concat([result[0] for result in results])
            #     attention_masks_with_label_name = ops.concat([result[1] for result in results])
            #     label_name_idx = ops.concat([result[2] for result in results])
            #     assert len(input_ids_with_label_name) > 0, "No label names appear in corpus!"
            #     label_name_data = {"input_ids": input_ids_with_label_name, "attention_masks": attention_masks_with_label_name, "labels": label_name_idx}
            #return data, label_name_data
        else:
            return data
    
    # find label name indices and replace out-of-vocab label names with [MASK]
    def label_name_in_doc(self, doc):
        doc = self.tokenizer.tokenize(doc)
        label_idx = -1 * ops.ones(self.max_len,type=mindspore.dtype.int32)
        new_doc = []
        wordpcs = []
        idx = 1 # index starts at 1 due to [CLS] token
        for i, wordpc in enumerate(doc):
            wordpcs.append(wordpc[2:] if wordpc.startswith("##") else wordpc)
            if idx >= self.max_len - 1: # last index will be [SEP] token
                break
            if i == len(doc) - 1 or not doc[i+1].startswith("##"):
                word = ''.join(wordpcs)
                flag = 0
                if word in self.label2class:
                    ##chenhu
                    if word in self.multi_class_names:
                        multi_class = self.multi_class_names.get(word)
                        for multi in multi_class:
                            if multi in doc:
                                label_idx[idx] = self.label2class[multi]
                                flag = 1
                        if flag == 0:
                            label_idx[idx] = self.label2class[word]
                    else:
                        label_idx[idx] = self.label2class[word]
                    # replace label names that are not in tokenizer's vocabulary with the [MASK] token
                    if word not in self.vocab:
                        wordpcs = [self.tokenizer.mask_token]
                new_word = ''.join(wordpcs)
                if new_word != self.tokenizer.unk_token:
                    idx += len(wordpcs)
                    new_doc.append(new_word)
                wordpcs = []
        if (label_idx >= 0).any():
            return ' '.join(new_doc), label_idx
        else:
            return None

    # find label name occurrences in the corpus
    def label_name_occurrence(self, docs):
        text_with_label = []
        label_name_idx = []
        for doc in docs:
            result = self.label_name_in_doc(doc)
            if result is not None:
                text_with_label.append(result[0])
                label_name_idx.append(ops.unsqueeze(result[1],dim=0))
        if len(text_with_label) > 0:
            encoded_dict = self.tokenizer.batch_encode_plus(text_with_label, add_special_tokens=True, max_length=self.max_len, 
                                                            padding='max_length', return_attention_mask=True, truncation=True, return_tensors='pt')
            input_ids_with_label_name = mindspore.Tensor(encoded_dict['input_ids'].numpy(),dtype=mindspore.dtype.int32)
            attention_masks_with_label_name = mindspore.Tensor(encoded_dict['attention_mask'].numpy(),dtype=mindspore.dtype.int32)
            label_name_idx = ops.concat(label_name_idx, axis=0)
        else:
            input_ids_with_label_name = ops.ones(self.max_len,type=mindspore.dtype.int32)
            attention_masks_with_label_name = ops.ones(self.max_len,type=mindspore.dtype.int32)
            label_name_idx = ops.ones(self.max_len,type=mindspore.dtype.int32)
        return input_ids_with_label_name, attention_masks_with_label_name, label_name_idx

    # read text corpus and labels from files
    def read_data(self, dataset_dir, train_file, test_file, test_label_file,web_file,web_label_file):
        # self.train_data, self.label_name_data = self.create_dataset(dataset_dir, train_file, None, "train.pt",
        #                                                             find_label_name=True, label_name_loader_name="label_name_data.pt")
        if test_file is not None:
            self.test_data = self.create_dataset(dataset_dir, test_file, test_label_file, "test.pt")
        #chenhu
        if web_file is not None:
            self.web_data = self.create_dataset(dataset_dir, web_file, web_label_file, "web.pt")

    # read label names from file
    def read_label_names(self, dataset_dir, label_name_file):
        label_name_file = open(os.path.join(dataset_dir, label_name_file))
        label_names = label_name_file.readlines()
        self.label_name_dict = {i: [word.lower() for word in category_words.strip().split()] for i, category_words in enumerate(label_names)}
        print(f"Label names used for each class are: {self.label_name_dict}")
        self.label2class = {}
        self.all_label_name_ids = [self.mask_id]
        self.all_label_names = [self.tokenizer._mask_token]
        self.multi_class_names={"":[]}
        for class_idx in self.label_name_dict:
            for word in self.label_name_dict[class_idx]:
                if word in self.label2class:
                    continue
                #assert word not in self.label2class, f"\"{word}\" used as the label name by multiple classes!"
                self.label2class[word] = class_idx
                if word in self.vocab:
                    self.all_label_name_ids.append(self.vocab[word])
                    self.all_label_names.append(word)
        #chenhu
        words_count = {'':[]}
        for class_idx in self.label_name_dict:
            for word in self.label_name_dict[class_idx]:
                if word not in words_count.keys():
                    words_count[word] = []
                words_count[word].append(class_idx)
        print(words_count)
        for key in words_count:
            if len(words_count[key]) > 1:
                if key not in self.multi_class_names:
                    self.multi_class_names[key] = []
                for idx in words_count[key]:
                    classes = self.label_name_dict[idx]
                    for cls in classes:
                        if cls != key:
                            self.multi_class_names[key].append(cls)
        print(self.label2class)
        print(self.multi_class_names)

    # create dataset loader
    def make_dataloader(self, rank, data_dict, batch_size):
        # if "labels" in data_dict:
        #     dataset = ds.GeneratorDataset(source=TrainDataset(input_ids=data_dict["input_ids"], attention_masks=data_dict["attention_masks"], labels=data_dict["labels"]),column_names=["input_ids","attention_masks","labels"])
        # else:
        #     dataset = ds.GeneratorDataset(source=TrainDataset(input_ids=data_dict["input_ids"], attention_masks=data_dict["attention_masks"]),column_names=["input_ids","attention_maks"])
        # dataset = dataset.batch(batch_size=batch_size)
        # dataset_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)
        return data_dict
    def make_mcp_dataloader(self, rank, data_dict, batch_size):
        #sampler = RandomSampler(num_shards=self.world_size, shard_id=rank)
        if "labels" in data_dict:
            dataset = ds.GeneratorDataset(source=TrainDataset(input_ids=data_dict["input_ids"], attention_masks=data_dict["attention_masks"], labels=data_dict["labels"],weights=data_dict['weight']),column_names=["input_ids","attention_masks","labels","weight"])
            dataset = dataset.batch(batch_size=batch_size)
        else:
            dataset = ds.GeneratorDataset(source=TrainDataset(input_ids=data_dict["input_ids"], attention_masks=data_dict["attention_masks"],weights=data_dict['weight']),column_names=["input_ids","attention_maks","weight"])
            dataset = dataset.batch(batch_size=batch_size)
        # sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank)
        # dataset_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)
        return dataset
    # filter out stop words and words in multiple categories
    def filter_keywords(self, category_vocab_size=200):
        all_words = defaultdict(list)
        sorted_dicts = {}
        for i, cat_dict in self.category_words_freq.items():
            sorted_dict = {k:v for k, v in sorted(cat_dict.items(), key=lambda item: item[1], reverse=True)[:category_vocab_size]}
            sorted_dicts[i] = sorted_dict
            for word_id in sorted_dict:
                all_words[word_id].append(i)
        repeat_words = []
        for word_id in all_words:
            if len(all_words[word_id]) > 1:
                repeat_words.append(word_id)
        self.category_vocab = {}
        for i, sorted_dict in sorted_dicts.items():
            self.category_vocab[i] = np.array(list(sorted_dict.keys()))
        stopwords_vocab = stopwords.words('english')
        for i, word_list in self.category_vocab.items():
            delete_idx = []
            for j, word_id in enumerate(word_list):
                word = self.inv_vocab[word_id]
                if word in self.label_name_dict[i]:
                    continue
                if not word.isalpha() or len(word) == 1 or word in stopwords_vocab or word_id in repeat_words:
                    delete_idx.append(j)
            self.category_vocab[i] = np.delete(self.category_vocab[i], delete_idx)

    # construct category vocabulary (distributed function)
    def category_vocabulary_dist(self, rank, top_pred_num=50, loader_name="category_vocab.pt"):
        model = self.set_up_dist(rank)
        model.set_train(False)
        label_name_dataset_loader = self.make_dataloader(rank, self.label_name_data, self.eval_batch_size)
        category_words_freq = {i: defaultdict(float) for i in range(self.num_class)}
        wrap_label_name_dataset_loader = tqdm(label_name_dataset_loader.create_dict_iterator())
        for batch in wrap_label_name_dataset_loader:
                # with torch.no_grad():
                    ## 128 200
                    input_ids = batch[0].to(rank)
                    ## 128 200
                    input_mask = batch[1].to(rank)
                    ## 128 200
                    label_pos = batch[2].to(rank)
                    ## 128 200
                    match_idx = label_pos >= 0
                    ## 128 200 30522
                    predictions = model(input_ids,
                                        pred_mode="mlm",
                                        token_type_ids=None, 
                                        attention_mask=input_mask)
                    ## 137 50
                    ## predictions[match_idx] 137 30522

                    _, sorted_res = ops.TopK(predictions[match_idx], top_pred_num)
                    ## 137 这里137的意思是，从当前batch 128 200中找到了137个类别名称，并将这个137个词的前50个替代拿到
                    label_idx = label_pos[match_idx]
                    #chenhu
                    for i, word_list in enumerate(sorted_res):
                        for j, word_id in enumerate(word_list):
                            category_words_freq[label_idx[i].item()][word_id.item()] += 1
        save_file = os.path.join(self.temp_dir, f"{rank}_"+loader_name)
        torch.save(category_words_freq, save_file)

    # construct category vocabulary
    def category_vocabulary(self, top_pred_num=50, category_vocab_size=100, loader_name="category_vocab.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"Loading category vocabulary from {loader_file}")
            self.category_vocab = torch.load(loader_file)
        else:
            print("Contructing category vocabulary.")
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            self.category_vocabulary_dist(top_pred_num, 0,loader_name)
            gather_res = []
            for f in os.listdir(self.temp_dir):
                if f[-3:] == '.pt':
                    gather_res.append(torch.load(os.path.join(self.temp_dir, f)))
            assert len(gather_res) == self.world_size, "Number of saved files not equal to number of processes!"
            self.category_words_freq = {i: defaultdict(float) for i in range(self.num_class)}
            for i in range(self.num_class):
                for category_words_freq in gather_res:
                    for word_id, freq in category_words_freq[i].items():
                        self.category_words_freq[i][word_id] += freq
            self.filter_keywords(category_vocab_size)
            torch.save(self.category_vocab, loader_file)
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        for i, category_vocab in self.category_vocab.items():
            print(f"Class {i} category vocabulary: {[self.inv_vocab[w] for w in category_vocab]}\n")

    # prepare self supervision for masked category prediction (distributed function)
    #chenhu
    def prepare_mcp_dist(self, rank, top_pred_num=50, match_threshold=10, loader_name="mcp_train.pt"):
        model = self.set_up_dist(rank)
        model.set_train(False)
        train_dataset_loader = self.make_dataloader(rank, self.train_data, self.eval_batch_size)

        all_input_ids = []
        all_mask_label = []
        all_input_mask = []
        all_input_weight=[]
        category_doc_num = defaultdict(int)
        wrap_train_dataset_loader = tqdm(train_dataset_loader.create_dict_iterator()) if rank == 0 else train_dataset_loader
        for batch in wrap_train_dataset_loader:
                    input_ids = batch[0]
                    input_mask = batch[1]
                    predictions = model(input_ids,
                                        pred_mode="mlm",
                                        token_type_ids=None,
                                        attention_mask=input_mask)
                    _, sorted_res = ops.TopK(predictions, top_pred_num, dim=-1)
                    for i, category_vocab in self.category_vocab.items():
                        match_idx = ops.zeros_like(sorted_res).bool()
                        for word_id in category_vocab:
                            match_idx = (sorted_res == word_id) | match_idx
                        match_count = ops.cumsum(match_idx.int(), axis=-1)
                        #chenhu
                        valid_idx = (match_count > 0.4*len(category_vocab)) & (input_mask > 0)
                        weights_count = ops.div(match_count.float(),len(category_vocab))
                        #weights_count = weights_count[valid_idx]
                        #valid_idx = (match_count > match_threshold) & (input_mask > 0)
                        valid_doc = ops.cumsum(valid_idx, axis=-1) > 0
                        if valid_doc.any():
                            mask_label = -1 * ops.ones_like(input_ids)
                            mask_label[valid_idx] = i
                            all_input_ids.append(input_ids[valid_doc].cpu())
                            all_mask_label.append(mask_label[valid_doc].cpu())
                            all_input_mask.append(input_mask[valid_doc].cpu())
                            all_input_weight.append(weights_count[valid_doc].cpu())
                            category_doc_num[i] += valid_doc.int().sum().item()
        all_input_ids = ops.concat(all_input_ids, axis=0)
        all_mask_label = ops.concat(all_mask_label, axis=0)
        all_input_mask = ops.concat(all_input_mask, axis=0)
        all_input_weight = ops.concat(all_input_weight,axis=0)
        save_dict = {
                "all_input_ids": all_input_ids,
                "all_mask_label": all_mask_label,
                "all_input_mask": all_input_mask,
                "all_input_weight":all_input_weight,
                "category_doc_num": category_doc_num,
        }
        save_file = os.path.join(self.temp_dir, f"{rank}_"+loader_name)
        torch.save(save_dict, save_file)

    # prepare self supervision for masked category prediction
    #chenhu
    def prepare_mcp(self, top_pred_num=50, match_threshold=2, loader_name="mcp_train.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        save_file = os.path.join(self.dataset_dir, "keywords.pt")
        if os.path.exists(loader_file):
            print(f"Loading masked category prediction data from {loader_file}")
            self.mcp_data = torch.load(loader_file)
            input_ids = self.mcp_data["input_ids"].numpy()
            input_masks = self.mcp_data["attention_masks"].numpy()
            mask_labels = self.mcp_data["labels"].numpy()
            category_doc_num = [[] for i in range(self.num_class)]
            #chenhu
            for i in range(len(input_ids)):
                input_id = input_ids[i]
                input_mask = input_masks[i]
                mask_label = mask_labels[i]
                mask_pos = mask_label >= 0
                label = mask_label[mask_pos]
                for i in range(len(mask_label)):
                    if mask_pos[i] == True:
                        category_doc_num[mask_label[i]].append(self.get_dict_key(input_id[i]))
            for i in range(len(category_doc_num)):
                category_doc_num[i] = dict(Counter(category_doc_num[i]).most_common(40))
            all_words = {}
            for i in range(len(category_doc_num)):
                cuur_dict = category_doc_num[i]
                for k,v in cuur_dict.items():
                    if k not in all_words.keys():
                        all_words[k] = 0
                    all_words[k] = all_words[k] + 1

            repeat_words = []
            for word,num in all_words.items():
                if num != 1:
                    repeat_words.append(word)
            stopwords_vocab = stopwords.words('english')

            for i in range(len(category_doc_num)):
                cuur_dict = category_doc_num[i].copy()
                for word,num in category_doc_num[i].items():
                    if word in self.label_name_dict:
                        continue
                    if not word.isalpha() or len(word) == 1 or word in stopwords_vocab or word in repeat_words:
                        del cuur_dict[word]
                category_doc_num[i] = cuur_dict
            torch.save(category_doc_num,save_file)
            print("")
        else:
            loader_file = os.path.join(self.dataset_dir, loader_name)
            print("Preparing self supervision for masked category prediction.")
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            mp.spawn(self.prepare_mcp_dist, nprocs=self.world_size, args=(top_pred_num, match_threshold, loader_name))
            gather_res = []
            for f in os.listdir(self.temp_dir):
                if f[-3:] == '.pt':
                    gather_res.append(torch.load(os.path.join(self.temp_dir, f)))
            assert len(gather_res) == self.world_size, "Number of saved files not equal to number of processes!"
            all_input_ids = ops.concat([res["all_input_ids"] for res in gather_res], axis=0)
            all_mask_label = ops.concat([res["all_mask_label"] for res in gather_res], axis=0)
            all_input_mask = ops.concat([res["all_input_mask"] for res in gather_res], axis=0)
            all_input_weight = ops.concat([res["all_input_weight"] for res in gather_res], axis=0)
            category_doc_num = {i: 0 for i in range(self.num_class)}
            for i in category_doc_num:
                for res in gather_res:
                    if i in res["category_doc_num"]:
                        category_doc_num[i] += res["category_doc_num"][i]
            print(f"Number of documents with category indicative terms found for each category is: {category_doc_num}")
            self.mcp_data = {"input_ids": all_input_ids, "attention_masks": all_input_mask, "labels": all_mask_label,"weight":all_input_weight}
            torch.save(self.mcp_data, loader_file)
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            # for i in category_doc_num:
            #     #chenhu
            #     assert category_doc_num[i] > 10, f"Too few ({category_doc_num[i]}) documents with category indicative terms found for category {i}; " \
            #            "try to add more unlabeled documents to the training corpus (recommend) or reduce `--match_threshold` (not recommend)"
        print(f"There are totally {len(self.mcp_data['input_ids'])} documents with category indicative terms.")

    # masked category prediction (distributed function)
    def mcp_dist(self, rank, epochs=5, loader_name="mcp_train"):
        print('......')
        model = self.set_up_dist(rank)
        model.set_train()
        #mcp_dataset_loader = self.make_mcp_dataloader(rank, self.mcp_data, self.train_batch_size)
        loader_file = os.path.join(self.dataset_dir, "mcp_train.pt")
        mcp_data = torch.load(loader_file)
        web_dataset_loader = self.make_dataloader(rank, self.web_data, self.train_batch_size)
        data_size=int(len(web_dataset_loader["input_ids"]))
        mcp_data_size = int(len(mcp_data["input_ids"]))
        #web_dataset_loader = web_dataset_loader.create_dict_iterator()
        total_steps = data_size * epochs
        mindstone = list(range(20, total_steps, 20))
        learning_rates = [1e-5 * (0.5 ** i) for i in range(len(mindstone))]
        #lr = nn.piecewise_constant_lr(mindstone, learning_rates)
        optimizer = nn.Adam(params=model.get_parameters(), lr=1e-5, eps=1e-8)
       # KLDivLoss = nn.KLDivLoss(reduction='none')
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)

        #model_weights = model.trainable_params()





        for i in range(epochs):

                total_train_loss = 0
                # for i in tqdm(range(mcp_data_size)):
                #     input_ids = ops.unsqueeze(mindspore.Tensor(mcp_data["input_ids"][i].numpy()),dim=0)
                #
                #     input_mask = ops.unsqueeze(mindspore.Tensor(mcp_data["attention_masks"][i].numpy()),dim=0)
                #
                #     labels = mcp_data["labels"][i].numpy()
                #     mask_pos = labels >= 0
                #     labels = labels[mask_pos]
                #     labels = mindspore.Tensor(labels, mindspore.int32)
                #
                #     def forward_fn():
                #         logits = model(input_ids, pred_mode="mcp", attention_mask=input_mask)
                #
                #         logits = ops.squeeze(logits).reshape(1, -1)
                #         label = labels.reshape(-1)
                #         loss = self.GCE_loss(logits, label)
                #         return loss, logits
                #
                #     grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
                #     (loss, _), grads = grad_fn()
                #     loss = mindspore.ops.depend(loss,optimizer(grads))
                #
                #     loss = loss.sum()
                #     total_train_loss += loss
               #  if rank == 0:
               #      print(f"Epoch {i+1}:")
               #  wrap_mcp_dataset_loader = tqdm(mcp_dataset_loader) if rank == 0 else mcp_dataset_loader
               # # web_dataset_loader = tqdm(web_dataset_loader) if rank == 0 else web_dataset_loader
               #  #model.zero_grad()
               #  for j, batch in enumerate(wrap_mcp_dataset_loader):
               #
               #      input_ids = batch[0].to(rank)
               #      input_mask = batch[1].to(rank)
               #      labels = batch[2].to(rank)
               #      weights=batch[3].to(rank)
               #      mask_pos = labels >= 0
               #      #weights_valid = weights > 0.4
               #      labels = labels[mask_pos]
               #      weights = weights[mask_pos]
               #      assert len(labels) == len(weights)
               #      # mask out category indicative words
               #      input_ids[mask_pos] = self.mask_id
               #      loss = train_step(input_ids,
               #                     pred_mode="classification",
               #                     token_type_ids=None,
               #                     attention_mask=input_mask,mask_pos=mask_pos)
               #
               #      #logit_softmax  = log_softmax(logits.view(-1, self.num_class))
               #      # loss = self.mcp_loss(logits.view(-1, self.num_class), labels.view(-1)) / self.accum_steps
               #      # for i in range(len(loss)):
               #      #     loss[i] = (1-torch.pow(loss[i],0.7))/0.7
               #      loss = loss / self.accum_steps
               #  #     #chenhu
               #  #     #按权重对loss进行加权
               #  #     # softmax = nn.Softmax()
               #  #     loss = torch.sum(torch.mul(loss,(weights)))
               #  #     loss = torch.sum(loss)
               #      total_train_loss += loss.item()
               #  #     loss.backward()
               #  #     if (j+1) % self.accum_steps == 0:
               #  #         # Clip the norm of the gradients to 1.0.
               #  #         ops.clip_by_global_norm(model.parameters(), 1.0)
               #  #         optimizer.step()
               #  #         model.zero_grad()
                #chenhu
                for i in tqdm(range(data_size)):
                    input_ids = web_dataset_loader["input_ids"][i]
                    input_mask = web_dataset_loader["attention_masks"][i]
                    labels = web_dataset_loader["labels"][i]

                    labels = mindspore.Tensor(labels, mindspore.int32)
                    def forward_fn():
                        logits = model(input_ids, pred_mode="mcp", attention_mask=input_mask)

                        logits = ops.squeeze(logits).reshape(1, -1)
                        label = labels.reshape(-1)
                        loss = self.GCE_loss(logits, label)
                        return loss, logits

                    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
                    (loss, _), grads = grad_fn()
                    loss = mindspore.ops.depend(loss,optimizer(grads))

                    loss = loss.sum()
                    total_train_loss += loss


                avg_train_loss = total_train_loss / data_size


                print(f"Average training loss: {avg_train_loss}")

                loader_file = os.path.join(self.dataset_dir, loader_name)
                    #torch.save(model.module.state_dict(), loader_file)
                mindspore.save_checkpoint(model,loader_file)
                #self.write_no_split_results(loader_name="mcp_model.ckpt", out_file="mcp_out.txt")

        loader_file = os.path.join(self.dataset_dir, loader_name)
        mindspore.save_checkpoint(model,loader_file)

    # masked category prediction
    ##chenhu
    def mcp(self, top_pred_num=50, match_threshold=20, epochs=5, loader_name="mcp_model"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        # if os.path.exists(loader_file):
        #     print(f"\nLoading model trained via masked category prediction from {loader_file}")
        #
        # else:
            #self.prepare_mcp(top_pred_num, match_threshold)
        print(f"\nTraining model via masked category prediction.")
            #mindspore.dataset.Dataset.map(operations=[self.mcp_dist],num_parallel_workers=self.world_size,)
        self.mcp_dist(rank=0,epochs=epochs,loader_name=loader_name)
            #mp.spawn(self.mcp_dist, nprocs=self.world_size, args=(epochs, loader_name))
        self.model.load_state_dict(torch.load(loader_file))

    def inference(self, model, dataset_loader, rank, return_type):
        if return_type == "data":
            all_input_ids = []
            all_input_mask = []
            all_preds = []
        elif return_type == "acc":
            pred_labels = []
            truth_labels = []
        elif return_type == "pred":
            pred_labels = []
        model.set_train(False)
        data_size = int(len(dataset_loader["input_ids"]))
        try:
            for i in range(data_size):
                    input_ids = dataset_loader["input_ids"][i]
                    input_mask = dataset_loader["attention_masks"][i]
                    logits = model(input_ids,
                                   pred_mode="mcp",
                                   token_type_ids=None,
                                   attention_mask=input_mask)
                    #logits = logits[:, 0, :]
                    if return_type == "data":
                        all_input_ids.append(input_ids)
                        all_input_mask.append(input_mask)
                        all_preds.append(nn.Softmax(dim=-1)(logits))
                    elif return_type == "acc":
                        labels = dataset_loader["labels"][i]
                        pred_labels.append(ops.Argmax(logits, dim=-1).cpu())
                        truth_labels.append(labels)
                    elif return_type == "pred":
                        #chenhu
                        pred_labels.append(nn.Softmax(axis=-1)(logits))
                        #pred_labels.append(torch.argmax(logits, dim=-1).cpu())
            if return_type == "data":
                all_input_ids = ops.concat(all_input_ids, axis=0)
                all_input_mask = ops.concat(all_input_mask, axis=0)
                all_preds = ops.concat(all_preds, axis=0)
                return all_input_ids, all_input_mask, all_preds
            elif return_type == "acc":
                pred_labels = ops.concat(pred_labels, axis=0)
                truth_labels = ops.concat(truth_labels, axis=0)
                samples = len(truth_labels)
                acc = (pred_labels == truth_labels).float().sum() / samples
                return acc.to(rank)
            elif return_type == "pred":
                pred_labels = ops.concat(pred_labels, axis=0)
                return pred_labels
        except RuntimeError as err:
            self.cuda_mem_error(err, "eval", rank)
    
    # use trained model to make predictions on the test set
    def write_results(self, loader_name="final_model.pt", out_file="out.txt"):
        truths = open(os.path.join(self.dataset_dir, self.args.gold_label_file))
        #chenhu
        map_file = open(os.path.join(self.dataset_dir, self.args.map_file))
        gold_labels = [str(label.strip()) for label in truths.readlines()]
        mapper = [int(label.strip()) for label in map_file.readlines()]
        loader_file = os.path.join(self.dataset_dir, loader_name)
        assert os.path.exists(loader_file)
        print(f"\nLoading final model from {loader_file}")
        self.model.load_state_dict(torch.load(loader_file))
        self.model.to(0)
        sampler = SequentialSampler()
        test_set = ds.GeneratorDataset(self.test_data["input_ids"], self.test_data["attention_masks"],sampler=sampler)
        #test_dataset_loader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=self.eval_batch_size)
        pred_labels = self.inference(self.model, test_set, 0, return_type="pred")

        repr_prediction = np.argmax(pred_labels,axis=1)
        assert len(mapper) == len(pred_labels) == len(repr_prediction)
        map_pred={}
        map_repr={}
        for i in range(len(pred_labels)):
            map_id = mapper[i]
            if map_id not in map_pred:
                map_pred[map_id]=[]
                map_repr[map_id]=[]
            map_pred[map_id].append(pred_labels[i])
            map_repr[map_id].append(repr_prediction[i])

        #chenhu
        new_pred_labels = [[] for i in range(len(map_pred))]
        new_repr_prediction=[0 for i in range(len(map_repr))]
        assert len(new_pred_labels) == len(new_repr_prediction)==len(gold_labels)
        for k,v in map_pred.items():
            max_item=-100
            max_pred=-1
            for vi in v:
                if max(vi) > max_item:
                    max_item = max(vi)
                    max_pred = list(vi.numpy()).index(max(vi.numpy()))
                    new_pred_labels[k] = vi
                    new_repr_prediction[k]=max_pred
        save_file = os.path.join(self.dataset_dir, 'pre_labels.pt')
        # 保存数据
        val = torch.tensor([item.cpu().detach().numpy() for item in new_pred_labels]).cuda()
        torch.save(val, save_file)
        #chenhu
        score = 0
        big_count = 0
        big_MRR = 0
        prof_dict = defaultdict(lambda: [0.0, 0])
        for i in range(len(new_pred_labels)):
            index_list = list(np.argsort(-new_pred_labels[i]))
            curr_golds = [int(i) for i in gold_labels[i].split(" ")]
            ranks = np.zeros(self.num_class)
            for gold in curr_golds:
                gold_index = index_list.index(gold)
                ranks[gold_index] = 1
            score = score + ndcg_at_k(ranks, 1000)
        print("ndcg")
        print(score / len(new_pred_labels))
        for i in range(len(new_pred_labels)):
            index_list = list(np.argsort(-new_pred_labels[i]))
            curr_golds = [int(i) for i in gold_labels[i].split(" ")]
            for gold in curr_golds:
                gold_index = index_list.index(gold)
                imrr = 1.0 / (gold_index + 1)
                prof_dict[gold][0] += imrr
                prof_dict[gold][1] += 1
        for prof, stats in prof_dict.items():
            big_count += 1
            big_MRR += float(stats[0] / stats[1])
        print("mrr")
        print(big_MRR / big_count)
        true_num = 0

        for i in range(len(new_repr_prediction)):
            curr_golds = [int(i) for i in gold_labels[i].split(" ")]
            if new_repr_prediction[i] in curr_golds:
                true_num = true_num + 1
        print("acc")
        print(float(true_num / len(new_repr_prediction)))

        out_file = os.path.join(self.dataset_dir, out_file)
        print(f"Writing prediction results to {out_file}")
        f_out = open(out_file, 'w')
        for label in new_repr_prediction:
            f_out.write(str(label) + '\n')
    def write_no_split_results(self, loader_name="mcp_train.pt", out_file="out.txt"):
        truths = open(os.path.join(self.dataset_dir, self.args.gold_label_file))
        #chenhu
        #map_file = open(os.path.join(self.dataset_dir, self.args.map_file))
        gold_labels = [str(label.strip()) for label in truths.readlines()]
        #mapper = [int(label.strip()) for label in map_file.readlines()]
        loader_file = os.path.join(self.dataset_dir, loader_name)
        print(loader_file)
        assert os.path.exists(loader_file)
        print(f"\nLoading final model from {loader_file}")
        param_dict = load_checkpoint(loader_file)
        load_param_into_net(self.model,param_dict)
        # self.model.load_state_dict(torch.load(loader_file))
        # self.model.to(0)
        # sampler = SequentialSampler()
        # test_set = ds.GeneratorDataset(self.test_data["input_ids"], self.test_data["attention_masks"],sampler=sampler)

        #test_dataset_loader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=self.eval_batch_size)
        pred_labels = self.inference(self.model, self.test_data, 0, return_type="pred")

        repr_prediction = np.argmax(pred_labels,axis=1)
        #assert len(mapper) == len(pred_labels) == len(repr_prediction)
        # map_pred={}
        # map_repr={}
        # for i in range(len(pred_labels)):
        #     map_id = mapper[i]
        #     if map_id not in map_pred:
        #         map_pred[map_id]=[]
        #         map_repr[map_id]=[]
        #     map_pred[map_id].append(pred_labels[i])
        #     map_repr[map_id].append(repr_prediction[i])

        #chenhu
        # new_pred_labels = [[] for i in range(len(map_pred))]
        # new_repr_prediction=[0 for i in range(len(map_repr))]
        # assert len(new_pred_labels) == len(new_repr_prediction)==len(gold_labels)
        # for k,v in map_pred.items():
        #     max_item=-100
        #     max_pred=-1
        #     for vi in v:
        #         if max(vi) > max_item:
        #             max_item = max(vi)
        #             max_pred = list(vi.numpy()).index(max(vi.numpy()))
        #             new_pred_labels[k] = vi
        #             new_repr_prediction[k]=max_pred
        # save_file = os.path.join(self.dataset_dir, 'pre_labels.pt')
        # 保存数据
        # val = torch.tensor([item.cpu().detach().numpy() for item in new_pred_labels]).cuda()
        # torch.save(val, save_file)
        #chenhu
        score = 0
        big_count = 0
        big_MRR = 0
        prof_dict = defaultdict(lambda: [0.0, 0])
        for i in range(len(pred_labels)):
            index_list = list(np.argsort(-pred_labels[i]))
            curr_golds = [int(i) for i in gold_labels[i].split(" ")]
            ranks = np.zeros(self.num_class)
            for gold in curr_golds:
                gold_index = index_list.index(gold)
                ranks[gold_index] = 1
            score = score + ndcg_at_k(ranks, 1000)
        print("ndcg")
        print(score / len(pred_labels))
        for i in range(len(pred_labels)):
            index_list = list(np.argsort(-pred_labels[i]))
            curr_golds = [int(i) for i in gold_labels[i].split(" ")]
            for gold in curr_golds:
                gold_index = index_list.index(gold)
                imrr = 1.0 / (gold_index + 1)
                prof_dict[gold][0] += imrr
                prof_dict[gold][1] += 1
        for prof, stats in prof_dict.items():
            big_count += 1
            big_MRR += float(stats[0] / stats[1])
        print("mrr")
        print(big_MRR / big_count)
        true_num = 0

        for i in range(len(repr_prediction)):
            curr_golds = [int(i) for i in gold_labels[i].split(" ")]
            if repr_prediction[i] in curr_golds:
                true_num = true_num + 1
        print("acc")
        print(float(true_num / len(repr_prediction)))

        out_file = os.path.join(self.dataset_dir, out_file)
        print(f"Writing prediction results to {out_file}")
        f_out = open(out_file, 'w')
        for label in repr_prediction:
            f_out.write(str(label) + '\n')
    # print error message based on CUDA memory error
    def cuda_mem_error(self, err, mode, rank):
        if rank == 0:
            print(err)
            if "CUDA out of memory" in str(err):
                if mode == "eval":
                    print(f"Your GPUs can't hold the current batch size for evaluation, try to reduce `--eval_batch_size`, current: {self.eval_batch_size}")
                else:
                    print(f"Your GPUs can't hold the current batch size for training, try to reduce `--train_batch_size`, current: {self.train_batch_size}")
        sys.exit(1)
def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.
