import modal
stub = modal.Stub()

@stub.function(gpu = 'L4',timeout=1200)
def loadIndicTrans2(dataset_name):
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
    subprocess.run("bash install.sh", shell=True)
    # os.chdir("")
    print(os.getcwd())

    with open('importIndic.py', 'w') as file:
        file.write("""
try:
    import torch
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
    
    # ---------------------------------------------------------------------------
    #                              English to Hindi
    # ---------------------------------------------------------------------------
    en_sents = [
        "When I was young, I used to go to the park every day.",
        "He has many old books, which he inherited from his ancestors.",
        "I can't figure out how to solve my problem.",
        "She is very hardworking and intelligent, which is why she got all the good marks.",
        "We watched a new movie last week, which was very inspiring.",
        "If you had met me at that time, we would have gone out to eat.",
        "She went to the market with her sister to buy a new sari.",
        "Raj told me that he is going to his grandmother's house next month.",
        "All the kids were having fun at the party and were eating lots of sweets.",
        "My friend has invited me to his birthday party, and I will give him a gift.",
    ]
    src_lang, tgt_lang = "eng_Latn", "hin_Deva"
    hi_translations = batch_translate(en_sents, src_lang, tgt_lang, en_indic_model, en_indic_tokenizer, ip)
    
    # print(f"{src_lang} - {tgt_lang}")
    # for input_sentence, translation in zip(en_sents, hi_translations):
    #     print(f"{src_lang}: {input_sentence}")
    # print(f"{tgt_lang}: {translation}")
    print("{} - {}".format(src_lang, tgt_lang))
    for input_sentence, translation in zip(en_sents, hi_translations):
        print("{}: {}".format(src_lang, input_sentence))
        print("{}: {}".format(tgt_lang, translation))
    
except Exception as e:
    # Handle the exception
    print("An error occurred: {}".format(e))
        """)
    result = subprocess.run(['python', 'importIndic.py'], stdout=subprocess.PIPE)

    # Print the output
    print(result.stdout.decode('utf-8'))



@stub.local_entrypoint()
def main():
    # provide dataset name among ai2_arc, gsm8k, lukaemon/mmlu
    dataset_name = "lukaemon/mmlu"
    
    loadIndicTrans2.remote(dataset_name)

    


