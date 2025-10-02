import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import os

#from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

import json
import re




#Collect data
if __name__ == '__main__':
    

    #Define prompt
    def infer_party_from_text(text, model):

        # Define data structure for output using pydantic
        class PartyInference(BaseModel):
            party: str = Field(description="party: Democratic or Republican")
            confidence: int = Field(description="confidence: from 1 to 5")

        # Output parser
        def extract_json(text):
            text = "".join(text.split("Now, please classify the following text:")[1:])
            json_pattern = r'\{.*?\}'
            matches = re.findall(json_pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            return None

        # System and user messages for the prompt with examples
        messages = [
            ("system", """You are a helpful assistant. Based on the following debater's text, 
    infer whether the debater's position aligns more with the Republican Party or the Democratic Party.
    Provide the answer with the confidence level of your answer.
    You must respond only with valid JSON in the following format:

    {{
      "party": "Democratic" or "Republican",
      "confidence": integer between 1 and 5
    }}

    Examples:

    Text: "I believe in universal healthcare."
    Response:
    {{
      "party": "Democratic",
      "confidence": 4
    }}

    Text: "I support lower taxes for businesses."
    Response:
    {{
      "party": "Republican",
      "confidence": 5
    }}"""),
            ("user", "Now, please classify the following text: {input}"),
            ("system","")
        ]

        # Create the prompt using ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages(messages)

        # Creating the chain
        chain = prompt | model

        # Input text (User's text)
        input_text = text

        # Invoking the chain with the input
        response = chain.invoke({"input": input_text})

        #print(f"original response: {response}\n")

        torch.cuda.empty_cache()

        # Extract JSON from the model's output
        parsed_response = extract_json(response)
        if parsed_response:
            # Validate the parsed response using pydantic
            try:
                inference = PartyInference(**parsed_response)
                return inference.dict()
            except:
                pass  # Handle validation errors if needed

        # If parsing fails, return None or handle accordingly
        return {"party": np.nan, "confidence": np.nan}

    
    print('code is running')
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    

    #Setup environment path
    #os.environ["TRANSFORMERS_CACHE"] = "/data/sg/bl46/cache/"
    os.environ["HF_HOME"] = "/data/sg/bl46/cache/"

    with open("./credentials_HF.txt") as f:
        os.environ["LLAMA_API_KEY"] = f.read()

    #Hugging face login  
    from huggingface_hub import login        
    hf_token = os.environ["LLAMA_API_KEY"]
    login(hf_token)


    #Load dataset 
    #DDO data
    df_ddo = pd.read_pickle('../dat/Llama/df_gpt_result_combined.p')

    #reddit data
    df_con = pd.read_pickle('../dat/Llama/con_inference_result_(total).p')
    df_dem = pd.read_pickle('../dat/Llama/con_inference_result_(total).p')


    #Load model 
    #model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model_id = "meta-llama/Llama-3.2-3B-Instruct" 

    #set tokenizer to prevent possible error. 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.resize_token_embeddings(len(tokenizer))


    #Create the text generation pipeline
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        device=1,
    )

    # Update HuggingFacePipeline
    hf = HuggingFacePipeline(pipeline=text_gen_pipeline)
    
    
    
    #Collect data 
    sample = df_ddo

    true_labels = []
    party_list = []
    conf_list = []
    index_list = []

    for i in tqdm(range(len(sample))):

        true_labels.append(sample['party_short'].iloc[i])

        text = sample['short_text'].iloc[i]

        with torch.no_grad():
            response = infer_party_from_text(text, hf)

        #print("####Final output####:", response, response['party'], response['confidence'])
        party, confidence = response['party'], response['confidence']

        party_list.append(party)
        conf_list.append(confidence)
        index_list.append(sample.index[i])

        print(sample.index[i], sample['party_short'].iloc[i], party, confidence)

        # Delete variables to free memory
        del response
        torch.cuda.empty_cache()


    #aggregate result
    result = pd.DataFrame({'original_idx':index_list, 
                           'original_party':true_labels,
                         'inferred_party':party_list, 
                           'confidence':conf_list})
    
    result.to_pickle('../dat/Llama/llama_ddo_result_(Llama-3.2-3B-Instruct).p')


    print(result.head())












