from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# === Paths ===
model_path = r'/home2/testdev/sultan/llm/models/Llama31-8B-IT/'
adapter_model_path = r'/home2/testdev/sultan/llm/models/llama3-8b-sql-tuned/'

# === Load Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # for safety during generation

# === Load Base Model ===
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
)

# === Load LoRA Adapter ===
model = PeftModel.from_pretrained(
    base_model,
    adapter_model_path,
    device_map="auto",
)

# === Merge LoRA for faster inference (optional) ===
model = model.merge_and_unload()

# === Inference function ===
def generate_response(prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

# === Example Usage ===
if __name__ == "__main__":
    prompt = """### Instruction:
You are a SQL Server Expert you can generate response from Natural Language to SQL.
Tell me the SQL Query for the below prompt
Prompt: Tell me total number of faults logged by Bhabna
Context: Here is the table schema for you 
Table Name: fault_bookv3

column: fault_id (BIGINT) NOT NULL
Description: Unique ID for each fault.

column: fault_time (DATETIME) NOT NULL
Description: Timestamp indicating when the fault occurred.

column: fault_duration (VARCHAR[25]) NOT NULL
Description: Duration for which the fault persisted.

column: system_name (VARCHAR[50])
Description: High-level category of the system where the fault occurred. Examples include "Control System Infrastructure", "Indus-2 RF System", "Microtron System", "Others", etc.

column: device_name (VARCHAR[50])
Description: Specific part of the system affected by the fault. Examples include "Software", "Beam-filling", "Power Supply", etc.

column: fault_description (VARCHAR[2000])
Description: Detailed description of the fault.

column: persons_involved (VARCHAR[200])
Description: Names of individuals involved in resolving the fault.

column: action_taken (VARCHAR[500])
Description: Actions taken to resolve the fault.

column: faulty_system (VARCHAR[200])
Description: More granular name of the affected subsystem. Most values are NULL.

column: human_error (VARCHAR[200])
Description: Indicates whether the fault was due to human error. Accepts values like "yes", "no", or NULL.

column: document (IMAGE[2147483647])
Description: Image related to the problem area.

column: document_name (VARCHAR[50])
Description: Name of the related document, if any formal documentation was created.

column: fda_entry (VARCHAR[3]) NOT NULL
Description: Indicates whether the entry is part of the FDA process. Accepts values like "yes" or "no".

column: logged_by (VARCHAR[25])
Description: Person who logged the fault and its diagnosis.

column: log_time (DATETIME)
Description: Timestamp indicating when the fault entry was logged.

column: first_observation (VARCHAR[4000]) NOT NULL
Description: Explanation of how the fault was initially detected.

column: beam_affected (VARCHAR[3]) NOT NULL
Description: Indicates whether the fault affected the beam. Accepts values like "yes" or "no".

### Response:
"""

    response = generate_response(prompt)
    print("\nGenerated Response:\n")
    print(response)
