import logging
from llama_cpp import Llama
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import config
import logging

config.setup_logging()
logger = logging.getLogger(__name__)

CLASS_LABELS = config.CLASS_LABELS

class IntentClassifier():
    def __init__(self, model_path=config.INTENT_CLASSIFIER_MODEL, num_labels=5):
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
        
        logging.info(
            f"\n🔍 **Query Classification Summary**"
            f"\n➡️ Query: {query}"
            f"\n🪄 Rule-Based Label: {rule_label}"
            f"\n🤖 Transformer Label: {classifier_label}"
            f"\n📈 Confidence: {classifier_confidence:.2f}"
        )
        
        if classifier_confidence > 0.7:
            return classifier_label
        elif rule_label == classifier_label:
            return rule_label
        elif (rule_label != classifier_label) and (classifier_confidence > 0.5):
            return classifier_label
        else:
            return "Uncertain"

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