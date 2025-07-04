import logging
from llama_cpp import Llama
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# MODEL_PATH = r"C:\Users\Administrator\Desktop\chatbot\backend\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
# MODEL_PATH = r"C:\Users\Administrator\Desktop\chatbot\backend\models\DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf"
# MODEL_PATH = r"C:\Users\Administrator\Desktop\chatbot\backend\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
EMBEDDER_MODEL_PATH = r"C:\Users\Administrator\Desktop\chatbot\backend\models\allminilm"
INTENT_CLASSIFIER_MODEL = r"C:\Users\Administrator\Desktop\chatbot\backend\models\intent_classifier_model"

logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=r'C:\Users\Administrator\Desktop\chatbot\backend\codes\logs\intent_classifier.log',
        filemode='a'  # Append to the log file if it exists
    )

logging.info("Logging Enabled...")

CLASS_LABELS = {
    0: "INTENT1_REALTIME",
    1: "INTENT2_ANALYTICAL",
    2: "INTENT3_FAULTINFO",
    3: "INTENT4_DOMAININFO",
    4: "INTENT5_GENERALINFO"
}

# def load_model(model_path):
#     model = Llama(model_path=model_path, n_ctx=4048, verbose=False)
#     return model

class IntentClassifier():
    def __init__(self, model_path=INTENT_CLASSIFIER_MODEL, num_labels=5):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.classifier_model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        self.classifier_model.eval()
        
    def rule_based_classification(self, query):
        query = query.lower()
        realtime_keywords = ["current", "now", "real-time", "live", "status"]
        analytical_keywords = ["average", "per day", "per month", "per hour", "last week", "summarized", "trend", "uptime"]
        fault_keywords = ["fault", "not working", "failure"]
        domain_keywords = ["physics"]

        if any(kw in query for kw in realtime_keywords):
            return CLASS_LABELS[0]
        elif any(kw in query for kw in analytical_keywords):
            return CLASS_LABELS[1]
        elif any(kw in query for kw in fault_keywords):
            return CLASS_LABELS[2]
        elif any(kw in query for kw in domain_keywords):
            return CLASS_LABELS[3]
        else: 
            return CLASS_LABELS[4]


    def transformer_based_classification(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding = True)
        with torch.no_grad():
            outputs = self.classifier_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=1).item()
        return CLASS_LABELS[predicted_class], float(probs.max())

    def classify_query(self, query):
        rule_label = self.rule_based_classification(query)
        classifier_label, classifier_confidence = self.transformer_based_classification(query)
        logging.info(f"Rule Label={rule_label}\nClassifier Label={classifier_label} with confidence score={classifier_confidence}")
        if classifier_confidence > 0.7:
            return classifier_label
        elif rule_label==classifier_label:
            return rule_label
        elif (rule_label!=classifier_label) and (classifier_confidence>0.5):
            return classifier_label
        else:
            return "Uncertain"
    #LLM based approach
    '''

    prompt = (
        "Context: You are a strict classifier for a chatbot router.\n"
        "You must classify the below Query into exactly one of the following categories:\n"
        "- general_query\n"
        "- search_in_db\n"
        "- fault_related\n\n"
        "Respond with ONLY one of these labels. Do NOT explain your answer.\n\n"
        f"Query: {query}\nCategory:"
    )

    output = llm(prompt=prompt, temperature = 0.0)
    if isinstance(output, dict) and "choices" in output:
        result = output["choices"][0]["text"]
    else :
        result = output

    VALID_CATEGORIES = {"general_query", "accelerator_data(SQL)", "fault_related(RAG)"}
    # if result not in VALID_CATEGORIES:
    #     logging.warning(f"Invalid classification result: '{result}' for query: '{query}'")
    #     return "unknown"
    
    return result.strip().lower() #used to remove leading and trailing spaces
    '''

if __name__ == "__main__":
    '''
    "live_data": ["What is the current beam energy?", "Is the magnet on?", "Live beam status"]
    "historical_data": ["Average uptime last week", "Summarize beam usage", "Trends of beam power"]
    "general": ["Who discovered electrons?", "What is a qubit?", "Define entropy"]
    '''

    prompt_list = ["What is the current beam energy?", "Is the magnet on?", "Live beam status", "Average uptime last week", "Summarize beam usage", "Trends of beam power", "Who discovered electrons?", "What is a qubit?", "Define entropy"]
    # llm = load_model(MODEL_PATH)
    # for prompt in prompt_list:
    #     result = f'Prompt: {prompt} \nClassified Result: {intent_classifier(prompt, llm)}\n'
    #     logging.info(result)
    obj = IntentClassifier()
    for prompt in prompt_list:
        result = f'Prompt: {prompt} \nClassified Result: {obj.classify_query(prompt)}\n'
        logging.info(result)