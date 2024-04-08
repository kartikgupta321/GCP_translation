
try:
    from datasets import load_dataset
    dataset_name = 'ai2_arc'
    print(dataset_name)
    
    if(dataset_name == "ai2_arc"):
        possible_configs = [
        'ARC-Challenge',
        'ARC-Easy'
        ]
        # columns to translate
        columns = ['question','choices']
        # columns not to translate, to keep in converted dataset as is.
        columns_asis = ['id','answerKey']
    
    elif(dataset_name == "gsm8k"):
        possible_configs = [
        "main",
        "socratic"
        ]
        # columns to translate
        columns = ['question','answer']
        # columns not to translate, to keep in converted dataset as is.
        columns_asis = []
        
    elif(dataset_name == "lukaemon/mmlu"):
        possible_configs = [
        "high_school_european_history",
        "business_ethics",
        "clinical_knowledge",
        "medical_genetics",
        ]
        # columns to translate
        columns = ['input','A','B','C','D']
        # columns not to translate, to keep in converted dataset as is.
        columns_asis = ['target']
    print(dataset_name, possible_configs)
    
    dataset = []
    if(dataset_name == "ai2_arc"):
        for config in possible_configs:
            base_url = f"https://huggingface.co/api/datasets/allenai/ai2_arc/parquet/{config}"
            data_files = {"train": base_url + "/train/0.parquet","test":base_url + "/test/0.parquet", "validation": base_url + "/validation/0.parquet"}
            dataset_slice = load_dataset("parquet", data_files=data_files)
            dataset.append(dataset_slice)
    elif(dataset_name == "gsm8k"):
        for config in possible_configs:
            base_url = f"https://huggingface.co/api/datasets/gsm8k/parquet/{config}"
            data_files = {"train": base_url + "/train/0.parquet","test":base_url + "/test/0.parquet"}
            dataset_slice = load_dataset("parquet", data_files=data_files)
            dataset.append(dataset_slice)
    elif(dataset_name == "lukaemon/mmlu"):
        for config in possible_configs:
            dataset_slice = load_dataset(dataset_name, config,ignore_verifications=True)
            dataset.append(dataset_slice)
    print(dataset)

    
except Exception as e:
    # Handle the exception
    print('An error occurred:'+ str(e))
        