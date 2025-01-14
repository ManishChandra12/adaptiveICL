def get_config(args):
    save_preds = args['save_preds']
    prepare = args['prepare']
    oracle_split = args['oracle_split']
    mode = args['mode']
    dataset = args['dataset']
    method = args['method']
    static_split = args['static_split']
    model_name = args['model_name']
    single_precision = args['single_precision']
    K_max = args['K_max']
    gpu_id = args['gpu_id']
    fraction = args['fraction']
    labels_available = args['labels_available']
    if method == 'static':
        fraction = 1

    if dataset == 'sst2':
        train_datapath = 'data/SST2'
        test_datapath = 'data/SST2'
        classes = ['negative', 'positive']
        classes_in_data = ['0', '1']
        if K_max == 0:
            prompt_prefix = 'Your task is to judge whether the sentiment of a movie review is positive or negative.\n'
        else:
            prompt_prefix = 'Your task is to judge whether the sentiment of a movie review is positive or negative. Here are a few examples:\n'
        prompt_suffix = 'Sentiment: '
        batch_size = 10
        eval_steps = 50
        learning_rate = 5e-3
        num_epochs = 100
        per_device_train_batch_size = 256
        per_device_eval_batch_size = 256
    elif dataset == 'trec':
        train_datapath = 'data/trec'
        test_datapath = 'data/trec'
        classes = ['Abbreviation', 'Entity', 'Description', 'Human', 'Location', 'Number']  #['Acronym', 'Body', 'Narration', 'Person', 'Place', 'Figure']
        classes_in_data = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]
        if K_max == 0:
            prompt_prefix = 'For the given question, your task is to determine the type of the sought after answer. Choose one among abbreviation, entity, description, human, location and numeric.\n'
        else:
            prompt_prefix = 'For the given question, your task is to determine the type of the sought after answer. Choose one among abbreviation, entity, description, human, location and numeric value. Here are a few examples:\n'
        prompt_suffix = 'Answer Type: '
        batch_size = 10
        eval_steps = 50
        learning_rate = 5e-3
        num_epochs = 100
        per_device_train_batch_size = 256
        per_device_eval_batch_size = 256
    elif dataset == 'cola':
        train_datapath = 'data/CoLA/'
        test_datapath = 'data/CoLA'
        classes = ['false', 'true']
        classes_in_data = ["0", "1"]
        if K_max == 0:
            prompt_prefix = 'Your task is to determine whether the given sentence is grammatically correct.\n'
        else:
            prompt_prefix = 'Your task is to determine whether the given sentence is grammatically correct. Here are a few examples:\n'
        prompt_suffix = 'Hypothesis: the sentence is grammatical, true or false? '
        batch_size = 10
        eval_steps = 50
        learning_rate = 5e-3
        num_epochs = 100
        per_device_train_batch_size = 256
        per_device_eval_batch_size = 256
    elif dataset == 'rte':
        train_datapath = 'data/RTE'
        test_datapath = 'data/RTE'
        classes = ['false', 'true']
        classes_in_data = ['not_entailment', 'entailment']
        if K_max == 0:
            prompt_prefix = 'Your task is to determine whether the given premise entails the hypothesis.\n'
        else:
            prompt_prefix = 'Your task is to determine whether the given premise entails the hypothesis. Here are a few examples:\n'
        prompt_suffix = 'Question: Does the premise entail the hypothesis, true or false? Answer: '
        batch_size = 3
        eval_steps = 50
        learning_rate = 5e-3
        num_epochs = 100
        per_device_train_batch_size = 256
        per_device_eval_batch_size = 256

    embed, neigh, input_dim, output_dim = None, None, None, None
    if prepare or method == 'dynamic':
        mode = 'similar'
        batch_size = 1
        K_max = 10
        embed = True
        neigh = True
        input_dim = embed * 384 + neigh * (1+K_max) #1 #(len(classes) * K_max) #(1+K_max)#(1+len(classes)) #len(classes) #K_max
        output_dim = K_max + 1
    return train_datapath, test_datapath, method, mode, K_max, dataset, eval_steps, learning_rate, num_epochs, per_device_train_batch_size, per_device_eval_batch_size, gpu_id, model_name, classes, classes_in_data, prompt_prefix, prompt_suffix, batch_size, single_precision, prepare, fraction, static_split, oracle_split, save_preds, embed, neigh, input_dim, output_dim, labels_available

