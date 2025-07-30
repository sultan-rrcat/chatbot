import logging
import torch
import config
import logging
from peft import PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, BertTokenizer, BertForSequenceClassification

config.setup_logging()
logger = logging.getLogger(__name__)

CLASS_LABELS = config.CLASS_LABELS

class IntentClassifier():
    def __init__(self, mistral_base_model, mistral_tokenizer, tinyllama_base_model, tinyllama_tokenizer):
        
        self.bert_tokenizer = BertTokenizer.from_pretrained(config.BERT_INTENT_CLASSIFIER_MODEL)
        self.bert_classifier_model = BertForSequenceClassification.from_pretrained(config.BERT_INTENT_CLASSIFIER_MODEL)
        self.bert_classifier_model.eval()

        self.mistral_base_model = mistral_base_model
        self.mistral_tokenizer = mistral_tokenizer

        self.tinyllama_trained_model = PeftModel.from_pretrained(tinyllama_base_model, config.TINYLLAMA_LLM_CLASSIFIER_ADAPTER_PATH, device_map="auto")
        self.tinyllama_tokenizer = tinyllama_tokenizer

    def llm_based_classification(self, user_query, max_new_tokens=32, temperature=0.9, top_p=0.95):
        """
        Performs intent classification using a direct call to the model's generate method.
        This is more efficient for classification tasks than streaming.
        """
        try:
            device = next(self.mistral_base_model.parameters()).device
                  
            prompt = f"""<s>[INST] <<SYS>>
You are an expert intent classification assistant for a chatbot in an accelerator control system environment. Your task is to accurately classify user queries into one of the following four categories.

**Your only output should be the intent label.**

---

### **Intent Categories and Descriptions:**

* **INTENT1_FAULT_SQL:**
    * **Description:** Use this for queries that ask for specific, factual, or quantifiable information from a structured database (SQL). These are often "what," "how many," "list," "show me," or "give me" type questions that can be answered with a database query.
    * **Keywords:** count, list, show, find, what is the number, how many, retrieve, who, what is the status.

* **INTENT2_FAULT_RAG:**
    * **Description:** This intent is for queries that require a deeper, more explanatory answer about a fault. These are "how to," "what is," or "explain" questions that need more than a simple data lookup, often involving troubleshooting or RCA (Root Cause Analysis).
    * **Keywords:** how to fix, what is the cause, explain, troubleshoot, guide, why, what are the steps to resolve.

* **INTENT3_DOMAIN_RAG:**
    * **Description:** For questions about internal, domain-specific information related to RRCAT, Indus-1, Indus-2, and Indian accelerator control systems. This is for general knowledge about the systems, not specific fault data.
    * **Keywords:** tell me about, what is, explain, describe, information on, details about.

* **INTENT4_GENERAL_INFO:**
    * **Description:** A fallback for general chitchat or any question that does not fit into the other three categories. Use this for greetings, general knowledge questions, and queries unrelated to the accelerator domain.

---

### **Examples:**

**INTENT1_FAULT_SQL Examples:**
* "What is the total number of faults recorded?"
* "List the top 5 `system_name` with the most faults."
* "How many faults were classified as 'human_error'?"
* "Retrieve all faults that occurred on '2025-06-18'."

**INTENT2_FAULT_RAG Examples:**
* "How do I troubleshoot a 'beam loss' event in Indus-2?"
* "What are the common causes for a vacuum leak in the linac?"
* "Explain the procedure for recalibrating the RF power supply."
* "What are the steps to resolve a 'klystron failure' alarm?"

**INTENT3_DOMAIN_RAG Examples:**
* "Tell me about the architecture of the Indus-2 control system."
* "What is the purpose of the RRCAT facility?"
* "Explain the difference between Indus-1 and Indus-2."
* "Can you provide documentation on the timing system?"

**INTENT4_GENERAL_INFO Examples:**
* "Hello, how are you today?"
* "What's the weather like in Indore?"
* "Who won the last cricket world cup?"
* "Can you tell me a joke?"
<</SYS>>

User Query: {user_query}

[/INST]"""
            
            # Tokenize the input prompt
            inputs = self.mistral_tokenizer(prompt, return_tensors="pt").to(device)

            # Generate the output directly
            outputs = self.mistral_base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.mistral_tokenizer.pad_token_id
            )
            
            # Decode the generated tokens, skipping the prompt
            output_text = self.mistral_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

            return output_text

        except Exception as e:
            logger.error(f"Unexpected Error in LLM classification: {e}")
            return "ERROR_UNKNOWN"
    
    def trained_llm_based_classification(self, user_query, max_new_tokens = 32):
        try:
            system_prompt = """You are an intent classification assistant for an accelerator control system chatbot. You receive user queries and classify each into exactly one of the following four categories:

- INTENT1_FAULT_SQL: For queries that require real-time, analytical, or ordered data from the SQL database.
- INTENT2_FAULT_RAG: For queries that need detailed explanations of issues or troubleshooting guidance, using historical logs stored in a vector database.
- INTENT3_DOMAIN_RAG: For questions related to internal information about RRCAT, Indus-1, Indus-2, or Indian accelerator control systems.
- INTENT4_GENERAL_INFO: For all other questions not covered by the above categories.

Instruction:
1. If you are not 100% confident about INTENT1_FAULT_SQL, INTENT2_FAULT_RAG and INTENT3_DOMAIN_RAG use INTENT4_GENERAL_INFO to handle such queries.
2. If you are not 100% confident about any of the classes reply with 'Undefined'
Respond with only the correct intent label, and nothing else.
"""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
        ]

            # Generate prompt with proper generation trigger:
            try:
                device = next(self.tinyllama_trained_model.parameters()).device  # Detect model's device
                input_ids = self.tinyllama_tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True, # Crucial: tells the model it's starting a new assistant turn
                    return_tensors="pt"
                ).to(device)
                # inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tokenized input to model device
            except Exception as e:
                logger.error(f"Tokenization Error: {e}")
                return "ERROR_TOKENIZATION"

            try:
                outputs = self.tinyllama_trained_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.95,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tinyllama_tokenizer.eos_token_id,
                    pad_token_id=self.tinyllama_tokenizer.pad_token_id,
                )
            except Exception as e:
                logger.error(f"Model Generation Error: {e}")
                return "ERROR_GENERATION"

            try:
                response = self.tinyllama_tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            except Exception as e:
                logger.error(f"Decoding Error: {e}")
                return "ERROR_DECODING"
            return response

        except Exception as e:
            logger.error(f"Unexpected Error: {e}")
            return "ERROR_UNKNOWN"
        
    def bert_based_classification(self, user_query):
        inputs = self.bert_tokenizer(user_query, return_tensors="pt", truncation=True, padding = True)
        with torch.no_grad():
            outputs = self.bert_classifier_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=1).item()
        return CLASS_LABELS[predicted_class], float(probs.max())
        
    def classify_query(self, query):
        logging.info(f"Starting classification for query: {query}")

        llm_label_clean = None
        trained_llm_label_clean = None
        bert_label = None
        bert_confidence = 0.0

        # LLM based classification
        try:
            llm_label = self.llm_based_classification(query)
            llm_label_clean = llm_label.strip().split()[0]  # handle extra tokens
            logging.info(f"LLM Label: {llm_label_clean}")
        except Exception as e:
            logging.error(f"LLM classification failed: {e}")

        # Trined LLM-based classification
        try:
            trained_llm_label = self.trained_llm_based_classification(query)
            trained_llm_label_clean = trained_llm_label.strip().split()[0]  # handle extra tokens
            logging.info(f"Trained LLM Label: {trained_llm_label_clean}")
        except Exception as e:
            logging.error(f"Trained LLM classification failed: {e}")

        # BERT-based classification
        try:
            bert_label, bert_confidence = self.bert_based_classification(query)
            logging.info(f"BERT Label: {bert_label} | Confidence: {bert_confidence:.2f}")
        except Exception as e:
            logging.error(f"BERT classification failed: {e}")

        intent_result = {}
        intent_result['mistral_intent'] = llm_label_clean
        intent_result['tinyllama_intent'] = trained_llm_label_clean
        intent_result['bert_intent'] = bert_label

        # Final decision logic
        if (llm_label_clean in CLASS_LABELS.values()
            or (trained_llm_label_clean in CLASS_LABELS.values() 
            and bert_label in CLASS_LABELS.values())):

            if((llm_label_clean == trained_llm_label_clean == bert_label)
               or (llm_label_clean == trained_llm_label_clean)
               or (llm_label_clean == bert_label)):
                logging.info(f"LLM, Trained LLM and BERT agreed or at least two are matching. Returning label: {llm_label_clean}")
                intent_result['classified_intent'] = llm_label_clean
            
            elif(trained_llm_label_clean == bert_label): # bert_confidence > 0.94
                intent_result['classified_intent'] = trained_llm_label_clean
            else:
                logging.warning("LLM and BERT classification mismatched or confidence too low. Returning fallback level 3.")
                intent_result['classified_intent'] = llm_label_clean
        else:
            intent_result['classified_intent'] = CLASS_LABELS[3]
        
        return intent_result

if __name__ == "__main__":
    prompt = 'list recent issues in Indus2 RF cavity'
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    mistral_base_model_path = config.MISTRAL_BASE_MODEL_PATH
    tinyllama_base_model_path = config.TINYLLAMA_BASE_MODEL_PATH
    
    mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_base_model_path)
    mistral_tokenizer.pad_token = mistral_tokenizer.eos_token

    tinyllama_tokenizer = AutoTokenizer.from_pretrained(tinyllama_base_model_path)
    tinyllama_tokenizer.pad_token = tinyllama_tokenizer.eos_token

    mistral_base_model = AutoModelForCausalLM.from_pretrained(
        mistral_base_model_path,
        quantization_config = bnb_config,
        device_map = "cuda"
    )
    mistral_base_model.to("cuda")

    tinyllama_base_model = AutoModelForCausalLM.from_pretrained(
        tinyllama_base_model_path,
        quantization_config=bnb_config if torch.cuda.is_available() else None, # Only quantize on GPU
        device_map="cuda" # Automatically map model layers to available devices
    )
    tinyllama_base_model.to("cuda")

    # ==============================================
    obj = IntentClassifier(mistral_base_model, mistral_tokenizer, tinyllama_base_model, tinyllama_tokenizer)
    result = f'Prompt: {prompt} \nClassified Result: {obj.classify_query(prompt)}\n'
    print(result)
