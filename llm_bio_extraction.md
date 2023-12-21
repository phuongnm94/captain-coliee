### Prompting query with Llama2

1. Request jaist computing server having 2 GPU A40 with interactive mode. 
   ```qsub -q GPU-S -l ngpus=2 -I ```

2. Access to the computing server and install env
    ```cmd
    conda create --name env_llm --file requirements_lm.txt
    conda activate env_llm  
    pip install --upgrade huggingface_hub
    ```
3. Following this doc to get access key on huggingface for Llama2 model: https://huggingface.co/meta-llama/Llama-2-70b-hf, or use my private key if u just want to try small times. 
   ```cmd
   huggingface-cli login  --token   <YOUR_PRIVATE_TOKEN>
   ```
4. Access to the computing server and run this code
    ```python  
    from transformers import LlamaTokenizer, AutoModel, AutoTokenizer,LlamaForCausalLM

    print("Loading model ...")
    model_name = 'meta-llama/Llama-2-70b-chat-hf'   # trained with chat and instruction  
    # model_name = 'meta-llama/Llama-2-70b-hf'  #  standard model 

    tokenizer = LlamaTokenizer.from_pretrained(model_name) 
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
    model.eval()

    prompt_content = """In bellow conversation between MICHAEL and LINDA: 
    "
    MICHAEL: What time is it?  They're supposed to run around midnight.  This is great isn't it?  Look at this night we've got here.  It couldn't be better.
    LINDA: Fine.
    MICHAEL: Is that, is that-that's just foam isn't it?  I can't even tell
    MICHAEL: Oh, no you know what I did?  I forgot my flashlight.  How could I be so stupid I forgot my flashlight.
    MICHAEL: The flashlight, the silver one.  There's only one isn't there?
    LINDA: It's not yours.
    MICHAEL: Well sure.
    MICHAEL: No of course our flashlight, yours and mine, mi flashlight es su flashlight naturally.  How could we not think to bring it?
    "

    What do you think about the characteristics and emotions of MICHAEL from the above conversation?"""

    with torch.no_grad():
        inputs = tokenizer(prompt_content, return_tensors="pt")
        input_ids = inputs['input_ids'].to("cuda")
        outputs = model.generate(input_ids, max_new_tokens=200)
        output_text = tokenizer.decode(outputs[0])
        print(output_text)
    ```
