import os
import random
import argparse
import datetime
import numpy as np
import json
import copy
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer


def arg_parser():
    parser = argparse.ArgumentParser(description='Adaptive ICL')
    parser.add_argument('--save_preds', action='store_true', help='Save final predictions')
    parser.add_argument('--no_save_preds', dest='save_preds', action='store_false')
    parser.set_defaults(save_preds=False)
    parser.add_argument('--prepare', action='store_true', help='Prepare data for training k-predictor model')
    parser.add_argument('--no_prepare', dest='prepare', action='store_false')
    parser.set_defaults(prepare=False)
    parser.add_argument('--oracle_split', type=str, choices=['train', 'test'], default='test',
                        help='Set to train if preparing data to train k-predictor, and to test for getting results with oracle settings',
                        required=False)
    parser.add_argument('--mode', type=str, choices=['random', 'similar'], help='How to select demonstrations', default='similar', required=False)
    parser.add_argument('--dataset', type=str, choices=['sst2', 'trec', 'rte', 'cola'],  help='Dataset', required=True)
    parser.add_argument('--method', type=str, choices=['dynamic', 'static'], help='Method', required=True)
    parser.add_argument('--static_split', type=str, choices=['train', 'dev', 'test'], default='test',
                        help='Use dev to identify best k and test for final evaluation', required=False)
    parser.add_argument('--model_name', type=str,
                        choices=['microsoft/phi-2', 'meta-llama/llama-2-7b-hf', 'meta-llama/llama-2-13b-hf'],
                        help='Model name', required=True)
    parser.add_argument('--single_precision', action='store_true')
    parser.add_argument('--no_single_precision', dest='subset_exists', action='store_false')
    parser.set_defaults(single_precision=True)
    parser.add_argument('--K_max', type=int, help='Max value of k', default=10)
    parser.add_argument('--gpu_id', type=int, help='GPU Id', default=0, required=False)
    parser.add_argument('--fraction', type=float,
                        help='Fraction of training data to use to generate training data for k-predictor model',
                        default=1, required=False)
    parser.add_argument('--cache_dir', type=str, help='Cache dir for storing transformers models',
                        default="/scratch/manish/hf_cache/", required=False)
    parser.add_argument('--labels_available', action='store_true')
    parser.add_argument('--no_labels_available', dest='labels_available', action='store_false')
    parser.set_defaults(labels_available=True)
    args = vars(parser.parse_args())
    return args

def get_kdtree(dataset):
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load SentenceTransformer model
    train_sentences = [ex.text_a if ex.text_b == "" else ex.text_a + " " + ex.text_b for ex in dataset['train']]  # Get sentence embeddings
    train_embeddings = sbert_model.encode(train_sentences)
    tree = KDTree(train_embeddings)
    return sbert_model, tree

def load_model_tokenizer(model_name, single_precision, cachedir):
    if model_name == 'meta-llama/llama-2-7b-hf' or model_name == 'meta-llama/llama-2-13b-hf':
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if single_precision else torch.float32, cache_dir=cachedir)
        tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side="left", cache_dir=cachedir)
    elif model_name == 'microsoft/phi-2':
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if single_precision else torch.float32, cache_dir=cachedir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", cache_dir=cachedir)
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

class InputExample(object):
    """A raw input example consisting of segments of text,
    a label for classification task or a target sequence of generation task.
    Other desired information can be passed via meta.

    Args:
        guid (:obj:`str`, optional): A unique identifier of the example.
        text_a (:obj:`str`, optional): The placeholder for sequence of text.
        text_b (:obj:`str`, optional): A secend sequence of text, which is not always necessary.
        label (:obj:`int`, optional): The label id of the example in classification task.
    """

    def __init__(self,
                 guid = None,
                 text_a = "",
                 text_b = "",
                 label = None,
                ):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        r"""Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        r"""Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SST2Processor():
    def __init__(self, classes_in_data):
        self.labels = classes_in_data
        self.label_mapping = {k: i for (i, k) in enumerate(self.labels)}

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[0:]):
                linelist = line.strip().split('\t')
                text_a = linelist[0]
                label = linelist[1]
                guid = "%s-%s" % (split, idx)
                example = InputExample(guid=guid, text_a=text_a, label=self.label_mapping[label])
                examples.append(example)
        return examples

class TrecProcessor():
    def __init__(self, classes_in_data):
        self.labels = classes_in_data
        self.label_mapping = {k: i for (i, k) in enumerate(self.labels)}

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.txt")
        examples = []
        with open(path, 'rb') as f:
            for idx, line in enumerate(f):
                fine_label, _, text = line.replace(b"\xf0", b" ").strip().decode().partition(" ")
                coarse_label = fine_label.split(":")[0]
                guid = "%s-%s" % (split, idx)
                text = text.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
                example = InputExample(guid=guid, text_a=text, label=self.label_mapping[coarse_label])
                examples.append(example)
        return examples

class colaProcessor():
    def __init__(self, classes_in_data):
        self.labels = classes_in_data
        self.label_mapping = {k: i for (i, k) in enumerate(self.labels)}

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                _, label, _, text = line.strip().split("\t")
                guid = "%s-%s" % (split, idx)
                example = InputExample(guid=guid, text_a=text, label=self.label_mapping[label])
                examples.append(example)
        return examples

class rteProcessor():
    def __init__(self, classes_in_data):
        self.labels = classes_in_data
        self.label_mapping = {k: i for (i, k) in enumerate(self.labels)}

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                if idx > 0:
                    _, premise, hypothesis, label = line.strip().split("\t")
                    guid = "%s-%s" % (split, idx)
                    example = InputExample(guid=guid, text_a=premise, text_b=hypothesis, label=self.label_mapping[label])
                    examples.append(example)
        return examples

def load_dataset(dataset, train_datapath, test_datapath, classes_in_data, tokenizer):
    dataset_dict = dict()
    if dataset == 'sst2':
        dataset_dict['train'] = SST2Processor(classes_in_data).get_examples(train_datapath, 'train')
        dataset_dict['test'] = SST2Processor(classes_in_data).get_examples(test_datapath, 'test')
        dataset_dict['dev'] = SST2Processor(classes_in_data).get_examples(test_datapath, 'dev')
    elif dataset == 'trec':
        train_val = TrecProcessor(classes_in_data).get_examples(train_datapath, 'train')
        dataset_dict['test'] = TrecProcessor(classes_in_data).get_examples(test_datapath, 'test')
        dataset_dict['train'], dataset_dict['dev'] = train_test_split(train_val, test_size=0.1, random_state=42, shuffle=True)
    elif dataset == 'cola':
        train_val = colaProcessor(classes_in_data).get_examples(train_datapath, 'train')
        dataset_dict['test'] = colaProcessor(classes_in_data).get_examples(test_datapath, 'dev')
        dataset_dict['train'], dataset_dict['dev'] = train_test_split(train_val, test_size=0.1, random_state=42, shuffle=True)
    elif dataset == 'rte':
        train_val = rteProcessor(classes_in_data).get_examples(train_datapath, 'train')
        dataset_dict['test'] = rteProcessor(classes_in_data).get_examples(test_datapath, 'dev')
        dataset_dict['train'], dataset_dict['dev'] = train_test_split(train_val, test_size=0.1, random_state=42, shuffle=True)

    train_labels = dict()
    for h in dataset_dict['train']:
        if h.label in train_labels.keys():
            train_labels[h.label] += 1
        else:
            train_labels[h.label] = 1
    print('Train labels count after splitting', train_labels)
    dev_labels = dict()
    for h in dataset_dict['dev']:
        if h.label in dev_labels.keys():
            dev_labels[h.label] += 1
        else:
            dev_labels[h.label] = 1
    print('Dev labels count after splitting', dev_labels)

    print("Length of train set: ", len(dataset_dict['train']))
    print("Length of test set", len(dataset_dict['test']))
    print("Length of dev set", len(dataset_dict['dev']))
    print("Train example at 0th  index: ", dataset_dict['train'][0])

    return dataset_dict

def pred_batch(dataset, prepare, prompt_prefix, tokenizer, splt, indexes, dataset_dict, k, max_rem_len, prompts, mode, tree, batch_size, sbert_model, model, class_idx, classes, device, labels_available):
    if dataset == 'sst2':
        enc = tokenizer.batch_encode_plus(
            [f'{prompt_prefix}Review: {test_example.text_a}\n' for test_example in dataset_dict[splt][indexes]],
            return_tensors='pt', padding='longest')
    elif dataset == 'trec':
        enc = tokenizer.batch_encode_plus(
            [f'{prompt_prefix}Question: {test_example.text_a}\n' for test_example in dataset_dict[splt][indexes]],
            return_tensors='pt', padding='longest')
    elif dataset == 'cola':
        enc = tokenizer.batch_encode_plus(
            [f'{prompt_prefix}Sentence: {test_example.text_a}\n' for test_example in dataset_dict[splt][indexes]],
            return_tensors='pt', padding='longest')
    elif dataset == 'rte':
        enc = tokenizer.batch_encode_plus(
            [f'{prompt_prefix}Premise: {test_example.text_a}\nHypothesis: {test_example.text_b}\n' for test_example in dataset_dict[splt][indexes]],
            return_tensors='pt', padding='longest')

    if k == 0:
        for key, enc_value in list(enc.items()):
            enc_value = enc_value[:, :max_rem_len]  # truncate any tokens that will not fit once the prompt is added
            enc[key] = torch.cat([enc_value, prompts[key][:enc_value.shape[0]]], dim=1)
    else:
        if mode == 'random':
            K_examples_all = [random.sample(dataset_dict['train'], k) for _ in range(batch_size)]
        elif mode == 'similar':
            test_sentences = [f'{test_example.text_a}' if test_example.text_b == "" else f'{test_example.text_a + " " + test_example.text_b}' for test_example in dataset_dict[splt][indexes]]
            test_embeddings = sbert_model.encode(test_sentences)  # Get the embedding of the test_sentence
            if prepare and splt == 'train':
                _, top_k_indices = tree.query(test_embeddings, k=k+1)  # find the top k most similar train sentences
                top_k_indices = [top_k_indices[0][1:]]
            else:
                _, top_k_indices = tree.query(test_embeddings, k=k)  # find the top k most similar train sentences
            K_examples_all = list()
            for i in top_k_indices:
                K_examples_all.append([dataset_dict['train'][j] for j in i])
        demonstrations = list()
        for K_examples in K_examples_all:
            if dataset == 'sst2':
                demonstrations.append(
                    ''.join([f'Review: {example.text_a}\nSentiment: {classes[example.label]}\n' for example in K_examples]) if labels_available else ''.join([f'Review: {example.text_a}\n' for example in K_examples]))
            elif dataset == 'trec':
                demonstrations.append(
                    ''.join([f'Question: {example.text_a}\nAnswer Type: {classes[example.label]}\n' for example in K_examples]) if labels_available else ''.join([f'Question: {example.text_a}\n' for example in K_examples]))
            elif dataset == 'cola':
                demonstrations.append(
                    ''.join([f'Sentence: {example.text_a}\nHypothesis: the sentence is grammatical, true or false? {classes[example.label]}\n' for example in K_examples]) if labels_available else ''.join([f'Sentence: {example.text_a}\n' for example in K_examples]))
            elif dataset == 'rte':
                demonstrations.append(
                    ''.join([f'Premise: {example.text_a}\nHypothesis: {example.text_b}\n Question: Does the premise entail the hypothesis, true or false? Answer: {classes[example.label]}\n' for example in K_examples]) if labels_available else ''.join([f'Premise: {example.text_a}\nHypothesis: {example.text_b}\n' for example in K_examples]))
        if dataset == 'sst2':
            enc = tokenizer.batch_encode_plus(
                [f'{prompt_prefix}{demonstrations[indx]}Review: {test_example.text_a}\n' for indx, test_example in
                 enumerate(dataset_dict[splt][indexes])], return_tensors='pt', padding='longest')
        elif dataset == 'trec':
            enc = tokenizer.batch_encode_plus(
                [f'{prompt_prefix}{demonstrations[indx]}Question: {test_example.text_a}\n' for indx, test_example in
                 enumerate(dataset_dict[splt][indexes])], return_tensors='pt', padding='longest')
        elif dataset == 'cola':
            enc = tokenizer.batch_encode_plus(
                [f'{prompt_prefix}{demonstrations[indx]}Sentence: {test_example.text_a}\n' for indx, test_example in
                 enumerate(dataset_dict[splt][indexes])], return_tensors='pt', padding='longest')
        elif dataset == 'rte':
            enc = tokenizer.batch_encode_plus(
                [f'{prompt_prefix}{demonstrations[indx]}Premise: {test_example.text_a}\nHypothesis: {test_example.text_b}\n' for indx, test_example in
                 enumerate(dataset_dict[splt][indexes])], return_tensors='pt', padding='longest')
        for key, enc_value in list(enc.items()):
            enc_value = enc_value[:, :max_rem_len]
            enc[key] = torch.cat([enc_value, prompts[key][:enc_value.shape[0]]], dim=1)
    seq_len = enc['input_ids'].shape[1]
    # print(enc)
    #print(tokenizer.batch_decode(enc['input_ids']))
    enc = {ky: v.to(device) for ky, v in enc.items()}
    with torch.no_grad():
        result = model(**enc).logits
    result = result[:, -1, class_idx]
    result = F.softmax(result, dim=1)
    labels = [test_example.label for test_example in dataset_dict[splt][indexes]]
    preds = torch.argmax(result, dim=-1)
    confidence = result[0][labels[0]].item()
    if prepare and splt == 'train' and (not labels_available):
        confidence = torch.max(result, dim=-1)[0].item()
    return seq_len, labels, preds, confidence


class MyDataset(Dataset):
    def __init__(self, df, num_classes, K_max, sbert_model, embed=True, neigh=True):
        self.df = df
        self.num_classes = num_classes
        self.K_max = K_max
        self.sbert_model = sbert_model
        self.embed = embed
        self.neigh = neigh
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        slic = self.df.iloc[[idx]]
        if self.embed:
            emb = slic.apply(lambda x: self.sbert_model.encode([x['text'].text_a if x['text'].text_b == "" else x['text'].text_a + " " + x['text'].text_b]), axis=1).values[0][0]
        else:
            emb = np.array([])
        if self.neigh:
            neighbourhood = list()
            for n in slic.neighbours.values[0]:
                neighbourhood.append(n.label / (self.num_classes * (self.num_classes-1) / 2))
            uniq, counts = np.unique(neighbourhood, return_counts=True)
            #neighbourhood = np.array([entropy(counts, base=self.num_classes)])
            neighbourhood.append(entropy(counts, base=self.num_classes))
            neighbourhood = np.array(neighbourhood)
        else:
            neighbourhood = np.array([])
        final_emb = np.concatenate([emb, neighbourhood])
        label = [0.0] * 11
        for kk in slic.k_work.values[0]:
            label[kk] = 1.0
        return final_emb, np.array(label)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.timestamp = int(datetime.datetime.now().timestamp())
        if not os.path.exists('models/' + str(self.timestamp)):
            os.makedirs('models/' + str(self.timestamp))

    def early_stop(self, validation_loss, model):
        print(self.min_validation_loss)
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(model.state_dict(), 'models/' + str(self.timestamp) + '/checkpoint.pt')
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        out = self.relu1(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def train_k_predictor(train, val, test, sbert_model, num_epochs, per_device_train_batch_size,
                      per_device_eval_batch_size, learning_rate, device, num_classes, K_max, embed, neigh,
                      input_dim, output_dim):
    train_dataset = MyDataset(train, num_classes, K_max, sbert_model, embed, neigh)
    val_dataset = MyDataset(val, num_classes, K_max, sbert_model, embed, neigh)
    test_dataset = MyDataset(test, num_classes, K_max, sbert_model, embed, neigh)
    train_data_loader = DataLoader(train_dataset, batch_size=per_device_train_batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=per_device_eval_batch_size, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=per_device_eval_batch_size, shuffle=False)

    model = Net(input_dim, output_dim)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    early_stopper = EarlyStopper(patience=3, min_delta=0)
    
    for epoch in range(num_epochs):
        all_train_loss = list()
        model.train()
        for i, (x, y) in enumerate(train_data_loader): 
            x = x.float().to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            all_train_loss.append(loss.item())
        model.eval()
        all_val_loss = list()
        for i, (x, y) in enumerate(val_data_loader):
            x = x.float().to(device)
            y = y.to(device)        
            outputs = model(x)
            val_loss = criterion(outputs, y)
            all_val_loss.append(val_loss.item())
        print('Epoch: {}. Train Loss: {}, Val Loss: {}.'.format(epoch+1, sum(all_train_loss)/len(all_train_loss) , sum(all_val_loss)/len(all_val_loss)))
        if early_stopper.early_stop(sum(all_val_loss)/len(all_val_loss), model):
            print('Early stopping')
            break
        else:
            print('Continuing')
    model.load_state_dict(torch.load('models/' + str(early_stopper.timestamp) + '/checkpoint.pt'))
    all_preds = list()
    all_labels = list()
    for i, (x, y) in enumerate(test_data_loader):
        x = x.float().to(device)
        y = y.to(device)
        outputs = model(x)
        outputs = F.sigmoid(outputs)
        preds = torch.argmax(outputs, dim=-1)
        all_preds.extend(preds)
        all_labels.extend(torch.argmax(y, 1))
    all_preds = [pred.item() for pred in all_preds]  #this
    unique, counts = np.unique(all_preds, return_counts=True)
    print(dict(zip(unique, counts)))
    all_labels = [l.item() for l in all_labels]
    report = classification_report(all_labels, all_preds, digits=4)
    print('Classification Report:')
    print(report)
    return model

def predict_ks(df, K_predictor_model, device, per_device_eval_batch_size, sbert_model, num_classes, K_max, embed, neigh, input_dim, output_dim):
    test_dataset = MyDataset(df, num_classes, K_max, sbert_model, embed, neigh)
    test_data_loader = DataLoader(test_dataset, batch_size=per_device_eval_batch_size, shuffle=False)
    all_preds = list()
    for i, (x, y) in enumerate(test_data_loader):
        x = x.float().to(device)
        y = y.to(device)
        outputs = K_predictor_model(x)
        outputs = F.sigmoid(outputs)
        preds = torch.argmax(outputs, dim=-1)
        all_preds.extend(preds)
    all_preds = [pred.item() for pred in all_preds]  #this
    return all_preds
