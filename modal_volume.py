import modal
stub = modal.Stub()

#create a volume to persist the translated data 
volume = modal.NetworkFileSystem.persisted("data")
MODEL_DIR = "/data"

# change parameters value according  time taken to translate
@stub.function( cpu=4, memory = 8276, gpu = 'A100', timeout=1200, network_file_systems={MODEL_DIR: volume})
def loadIndicTrans2(dataset_name):
    import time
    start_time = time.time()

    import os 
    import subprocess
    
    commands = [
    "pip install -q bitsandbytes",
    "apt update ", 
    "apt install -y git",
    "git clone https://github.com/AI4Bharat/IndicTrans2"
    ]
    for command in commands:
        subprocess.run(command, shell=True)

    os.chdir("IndicTrans2/huggingface_interface")
    subprocess.run(["bash", "install.sh"])


# worked around with creating a file, as changing directory was not working
    with open('importIndic.py', 'w') as file:
        file.write(f'''
try:
    import torch
    import subprocess
    subprocess.run(["apt-get", "update", "-y"])
    subprocess.run(["apt-get", "install", "wget", "-y"])
    from datasets import load_dataset
    import os
    import pandas as pd
    import csv
    print(torch.cuda.get_device_name(0))
    import sys
    from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
    from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
    
    en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"  # ai4bharat/indictrans2-en-indic-dist-200M
    
    BATCH_SIZE = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if len(sys.argv) > 1:
        quantization = sys.argv[1]
    else:
        quantization = ""
    
    
    def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
        if quantization == "4-bit":
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8-bit":
            qconfig = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
        else:
            qconfig = None
    
        tokenizer = IndicTransTokenizer(direction=direction)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=qconfig,
        )
    
        if qconfig == None:
            model = model.to(DEVICE)
            model.half()
        model.eval()
        return tokenizer, model
    
    def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
        translations = []
        for i in range(0, len(input_sentences), BATCH_SIZE):
            batch = input_sentences[i : i + BATCH_SIZE]
    
            # Preprocess the batch and extract entity mappings
            batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
    
            # Tokenize the batch and generate input encodings
            inputs = tokenizer(
                batch,
                src=True,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(DEVICE)
    
            # Generate translations using the model
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )
    
            # Decode the generated tokens into text
            generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)
    
            # Postprocess the translations, including entity replacement
            translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)
            del inputs
            torch.cuda.empty_cache()
        return translations
    
    
    ip = IndicProcessor(inference=True)
    en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic", quantization)

    
    dataset_name = '{dataset_name}'
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
        "high_school_us_history",
        "high_school_physics",
        "high_school_world_history",
        "virology",
        "high_school_microeconomics",
        "econometrics",
        "college_computer_science",
        "high_school_biology",
        "abstract_algebra",
        "professional_accounting",
        "philosophy",
        "professional_medicine",
        "nutrition",
        "global_facts",
        "machine_learning",
        "security_studies",
        "public_relations",
        "professional_psychology",
        "prehistory",
        "anatomy",
        "human_sexuality",
        "college_medicine",
        "high_school_government_and_politics",
        "college_chemistry",
        "logical_fallacies",
        "high_school_geography",
        "elementary_mathematics",
        "human_aging",
        "college_mathematics",
        "high_school_psychology",
        "formal_logic",
        "high_school_statistics",
        "international_law",
        "high_school_mathematics",
        "high_school_computer_science",
        "conceptual_physics",
        "miscellaneous",
        "high_school_chemistry",
        "marketing",
        "professional_law",
        "management",
        "college_physics",
        "jurisprudence",
        "world_religions",
        "sociology",
        "us_foreign_policy",
        "high_school_macroeconomics",
        "computer_security",
        "moral_scenarios",
        "moral_disputes",
        "electrical_engineering",
        "astronomy",
        "college_biology"
        ]
        # columns to translate
        columns = ['input','A','B','C','D']
        # columns not to translate, to keep in converted dataset as is.
        columns_asis = ['target']

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

    if(dataset_name=='lukaemon/mmlu'):
        os.makedirs("lukaemon_mmlu_files", exist_ok=True)
        os.chdir("lukaemon_mmlu_files")
    else:
        os.makedirs(dataset_name + "_files", exist_ok=True)
        os.chdir(dataset_name + "_files")
    
    for i in range(len(possible_configs)):
        for set in dataset[i]:
            set_list = []
            
            for col in columns:
                values = [str(item[col]) for item in dataset[i][set]]
                
                if __name__ == '__main__':
                    result =[]
                    result = batch_translate(values[:20], "eng_Latn", "hin_Deva", en_indic_model, en_indic_tokenizer, ip)
                set_list.append(result)
    
            # Create folders for each configuration
            current_directory = os.getcwd()
            
            # Specify the path of the 'config' folder
            config_folder_path = os.path.join(current_directory, possible_configs[i])
            
            # Create the 'config' folder
            os.makedirs(config_folder_path, exist_ok=True)       
        
            # Transpose the 2D list
            transposed_data = list(map(list, zip(*set_list)))
            
            # to add untranslated columns in dataset
            for row in range(len(transposed_data)):
                for col in columns_asis:
                    if col=='id':
                            position = 0
                    else:
                        position = len(transposed_data[row])
                    transposed_data[row].insert(position, dataset[i][set][col][row]) 
                
            path = os.path.join(possible_configs[i],  set+'.csv')
    
            # append to previosly created csv file in case full dataset was not converted
            with open(path, 'w', encoding='utf-8') as f:
                # using csv.writer method from CSV package
                write = csv.writer(f)
                # write.writerow(columns)
                write.writerows(transposed_data)

    
    import shutil

    # Specify the folder path
    source_folder = os.getcwd()
    
    # Specify the destination zip file path
    destination_zip = '/data/{dataset_name}.zip'

    # Create a zip file
    shutil.make_archive(destination_zip.replace('.zip', ''), 'zip', source_folder)

    
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
    # some configs of mmlu are commented, uncomment to translate all
    
    dataset_name = "ai2_arc"
    
    loadIndicTrans2.remote(dataset_name)