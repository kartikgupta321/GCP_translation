import modal
stub = modal.Stub()

#create a volume to persist the translated data 
volume = modal.NetworkFileSystem.persisted("data")
MODEL_DIR = "/data"

# change parameters value according  time taken to translate
@stub.function( cpu=2, memory = 4276, gpu = 'A10G', timeout=1200, network_file_systems={MODEL_DIR: volume})
def loadIndicTrans2(dataset_name):
    import time
    start_time = time.time()
    
    import os 
    import subprocess
    subprocess.run(["pip", "install", "datasets"])
    
    # worked around with creating a file, as changing directory was not working
    with open('importIndic.py', 'w') as file:
        file.write(f'''
try:
    from datasets import load_dataset
    import subprocess
    subprocess.run(["apt-get", "update", "-y"])
    subprocess.run(["apt-get", "install", "wget", "-y"])
    
    dataset_name = '{dataset_name}'
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
        
            for i in ['train','test','validation']:
                subprocess.run(["wget", f"https://huggingface.co/api/datasets/allenai/ai2_arc/parquet/{{config}}/{{i}}/0.parquet", "-O", f'{{config}}{{i}}.parquet'])
            
            data_files = {{"train": f'{{config}}train.parquet',"test":f'{{config}}test.parquet', "validation": f'{{config}}validation.parquet'}}
            dataset_slice = load_dataset("parquet", data_files=data_files)
            dataset.append(dataset_slice)
    elif(dataset_name == "gsm8k"):
        for config in possible_configs:
        
            for i in ['train','test']:
                            
                            subprocess.run(["wget", f"https://huggingface.co/api/datasets/gsm8k/parquet/{{config}}/{{i}}/0.parquet", "-O", f'{{config}}{{i}}.parquet'])
            
            data_files = {{"train": f'{{config}}train.parquet',"test":f'{{config}}test.parquet'}}
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
        ''')
    result = subprocess.run(['python', 'importIndic.py'], stdout=subprocess.PIPE)
    
    # Print the output
    print(result.stdout.decode('utf-8'))
    print(start_time - time.time())


@stub.local_entrypoint()
def main():
    # provide dataset name among ai2_arc, gsm8k, lukaemon/mmlu
    dataset_name = "lukaemon/mmlu"
    
    loadIndicTrans2.remote(dataset_name)