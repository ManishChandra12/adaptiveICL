import random
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from .config import get_config
from .utils import arg_parser, load_model_tokenizer, load_dataset, get_kdtree, pred_batch, train_k_predictor, predict_ks


def main():
    args = arg_parser()
    print(args)

    train_datapath, test_datapath, method, mode, K_max, dataset, eval_steps, learning_rate, num_epochs, per_device_train_batch_size, per_device_eval_batch_size, gpu_id, model_name, classes, classes_in_data, prompt_prefix, prompt_suffix, batch_size, single_precision, prepare, fraction, static_split, oracle_split, save_preds, embed, neigh, input_dim, output_dim, labels_available = get_config(args)

    seed_value = 42
    random.seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.cuda.set_device(gpu_id)
    device = torch.device('cuda:'+str(gpu_id) if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model, tokenizer = load_model_tokenizer(model_name, single_precision, args['cache_dir'])
    model.to(device)
    model.eval()
    class_idx = tuple([tokenizer.encode(clas, add_special_tokens=False)[0] for clas in classes])
    #print(class_idx)
    dataset_dict = load_dataset(dataset, train_datapath, test_datapath, classes_in_data, tokenizer)

    if mode == 'similar':
        sbert_model, tree = get_kdtree(dataset_dict)

    prompts = tokenizer.batch_encode_plus([prompt_suffix for _ in range(batch_size)], return_tensors='pt', padding='longest', add_special_tokens=False)
    #print(prompts)
    max_rem_len = model.config.max_position_embeddings - prompts['input_ids'].shape[1]

    if prepare:
        K_range = list(range(0, K_max+1))
        splt = oracle_split
        all_preds = list()
        all_labels = list()
        seq_lens = list()
        all_ks = list()
        all_data = list()
        for start_idx in tqdm(range(0, len(dataset_dict[splt]), batch_size)):
            best_confidence = -1
            best_k = None
            indexes = slice(start_idx, start_idx + batch_size)
            for k in K_range:
                seq_len, labels, preds, confidence = pred_batch(dataset, prepare, prompt_prefix, tokenizer, splt,
                                                                indexes, dataset_dict, k, max_rem_len, prompts, mode,
                                                                tree, batch_size, sbert_model, model, class_idx,
                                                                classes, device, labels_available)
                if labels_available and labels[0] == preds.item():
                    best_pred = preds
                    best_k = k
                    best_seq = seq_len
                    break
                else:
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_k = k
                        best_seq = seq_len
                        best_pred = preds

            k_work = list()
            if labels_available:
                for k in K_range:
                    seq_len, labels, preds, confidence = pred_batch(dataset, prepare, prompt_prefix, tokenizer, splt,
                                                                    indexes, dataset_dict, k, max_rem_len, prompts,
                                                                    mode, tree, batch_size, sbert_model, model,
                                                                    class_idx, classes, device, labels_available)
                    if labels[0] == preds.item():
                        k_work.append(k)
            else:
                k_work = [best_k]

            if len(k_work) > 0:
                assert best_k in k_work

            all_ks.append(best_k)
            seq_lens.append(best_seq)
            all_preds.extend(best_pred)
            all_labels.extend(labels)

            test_sentences = [f'{test_example.text_a}' if test_example.text_b == "" else f'{test_example.text_a + " " + test_example.text_b}' for test_example in dataset_dict[splt][indexes]]
            test_embeddings = sbert_model.encode(test_sentences)  # Get the embedding of the test_sentence
            top_k_dist, top_k_indices = tree.query(test_embeddings, k=K_max+1)  # find the top k most similar train sentences
            top_k_indices = [top_k_indices[0][1:]]
            top_k_dist = [top_k_dist[0][1:]]
            K_examples_all = list()
            for i in top_k_indices:
                K_examples_all.append([dataset_dict['train'][j] for j in i])
            this_ex = dataset_dict[splt][indexes]
            all_data.append([this_ex[0], K_examples_all[0], top_k_dist[0].tolist(), k_work, best_k])

        df = pd.DataFrame(all_data, columns=['text', 'neighbours', 'distances', 'k_work', 'best_k'])
        subdr = '' if labels_available else 'no_labels/'
        if splt == 'train':
            try:
                train, test = train_test_split(df, test_size=0.15, random_state=0, stratify=df[['best_k']])
                train, val = train_test_split(train, test_size=(1.5/8.5), random_state=0, stratify=train[['best_k']])
            except:
                train, test = train_test_split(df, test_size=0.15, random_state=0)
                train, val = train_test_split(train, test_size=(1.5/8.5), random_state=0)
            df.to_pickle(f"data/k_predictor/{subdr}{dataset}_{model_name.split('/')[1]}_{K_range[0]}.p")
            train.to_pickle(f"data/k_predictor/{subdr}{dataset}_{model_name.split('/')[1]}_{K_range[0]}_train.p")
            val.to_pickle(f"data/k_predictor/{subdr}{dataset}_{model_name.split('/')[1]}_{K_range[0]}_val.p")
            test.to_pickle(f"data/k_predictor/{subdr}{dataset}_{model_name.split('/')[1]}_{K_range[0]}_test.p")

            df = pd.read_pickle(f"data/k_predictor/{subdr}{dataset}_{model_name.split('/')[1]}_{K_range[0]}.p")

            print('Train set size: ', len(train))
            print('Val set size: ', len(val))
            print('Test set size: ', len(test))

            print(df.best_k.value_counts())
            print(train.best_k.value_counts())
            print(val.best_k.value_counts())
            print(test.best_k.value_counts())
        else:
            df.to_pickle(f"data/k_predictor/{subdr}actual_test_{dataset}_{model_name.split('/')[1]}_{K_range[0]}.p")
            print(df.best_k.value_counts())

        all_ks = np.array(all_ks)
        unique, counts = np.unique(all_ks, return_counts=True)
        print(dict(zip(unique, counts)))
        print("Mean k: ", all_ks.mean())
    else:
        all_preds = list()
        all_labels = list()
        seq_lens = list()
        if method == 'static':
            for start_idx in tqdm(range(0, len(dataset_dict[static_split]), batch_size)):
                indexes = slice(start_idx, start_idx + batch_size)
                seq_len, labels, preds, confidence = pred_batch(dataset, prepare, prompt_prefix, tokenizer,
                                                                static_split, indexes, dataset_dict, K_max, max_rem_len,
                                                                prompts, mode, tree, batch_size, sbert_model, model,
                                                                class_idx, classes, device, labels_available)
                seq_lens.append(seq_len)
                all_preds.extend(preds)
                all_labels.extend(labels)
        elif method == 'dynamic':
            subdr = '' if labels_available else 'no_labels/'
            train = pd.read_pickle(f"data/k_predictor/{subdr}{dataset}_{model_name.split('/')[1]}_0_train.p")
            val = pd.read_pickle(f"data/k_predictor/{subdr}{dataset}_{model_name.split('/')[1]}_0_val.p")
            test = pd.read_pickle(f"data/k_predictor/{subdr}{dataset}_{model_name.split('/')[1]}_0_test.p")

            train = train.sample(frac=fraction)#, random_state=42)
            val = val.sample(frac=fraction)#, random_state=42)
            test = test.sample(frac=fraction)#, random_state=42)
            print(len(train))
            K_predictor_model = train_k_predictor(train, val, test, sbert_model, num_epochs, per_device_train_batch_size, per_device_eval_batch_size, learning_rate, device, len(classes), K_max, embed, neigh, input_dim, output_dim)
            all_test_data = list()
            for start_idx in tqdm(range(0, len(dataset_dict['test']), 1)):
                indexes = slice(start_idx, start_idx + batch_size)
                test_sentences = [f'{test_example.text_a}' if test_example.text_b == "" else f'{test_example.text_a + " " + test_example.text_b}' for test_example in dataset_dict['test'][indexes]]
                test_embeddings = sbert_model.encode(test_sentences)  # Get the embedding of the test_sentence
                top_k_dist, top_k_indices = tree.query(test_embeddings, k=K_max)  # find the top k most similar train sentences
                K_examples_all = list()
                for i in top_k_indices:
                    K_examples_all.append([dataset_dict['train'][j] for j in i])
                this_ex = dataset_dict['test'][indexes]
                all_test_data.append([this_ex[0], K_examples_all[0],top_k_dist[0].tolist(), [1], 1])
            df = pd.DataFrame(all_test_data, columns=['text', 'neighbours', 'distances', 'k_work', 'best_k'])
            pred_k = predict_ks(df, K_predictor_model, device, per_device_eval_batch_size, sbert_model, len(classes), K_max, embed, neigh, input_dim, output_dim)
            pred_k = np.array(pred_k)
            pred_df = pd.DataFrame(pred_k, columns=['pred_k'])
            pred_df.to_pickle(f'./preds_{dataset}_0.p')
            
            for start_idx in tqdm(range(0, len(dataset_dict['test']), 1)):
                indexes = slice(start_idx, start_idx + batch_size)
                seq_len, labels, preds, confidence = pred_batch(dataset, prepare, prompt_prefix, tokenizer, 'test',
                                                                indexes, dataset_dict, pred_k[start_idx], max_rem_len,
                                                                prompts, mode, tree, 1, sbert_model, model, class_idx,
                                                                classes, device, labels_available)
                seq_lens.append(seq_len)
                all_preds.extend(preds)
                all_labels.extend(labels)

            unique, counts = np.unique(pred_k, return_counts=True)
            print(dict(zip(unique, counts)))
            print("Mean k: ", pred_k.mean())
    seq_lens = np.array(seq_lens)
    print("Mean Sequence length: ", seq_lens.mean())
    print("Min Sequence length: ", seq_lens.min())
    print("Max Sequence length: ", seq_lens.max())
    print("95th percentile: ", np.percentile(seq_lens, 95))
    print("99th percentile: ", np.percentile(seq_lens, 99))
    print("99.9th percentile: ", np.percentile(seq_lens, 99.9))
            
    all_preds = [pred.item() for pred in all_preds]
    #print(all_preds)
    #print(all_labels)
    report = classification_report(all_labels, all_preds, digits=4)
    print('Classification Report:')
    print(report)

    if save_preds:
        pred_lab = list(zip(all_preds, all_labels))
        df = pd.DataFrame(pred_lab, columns=['pred','label'])
        df.to_csv(f"./results/pred_label_{dataset}_{model_name.split('/')[1]}_{method}.csv", index=False)


if __name__ == "__main__":
    main()
