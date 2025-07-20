import os
import json
import warnings
import yaml
from tqdm import tqdm
from openai import OpenAI
warnings.filterwarnings("ignore")

config_path = os.path.join("config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

openai_api_key = config["openai"]["api_key"]
openai_organization = config["openai"]["organization"]

client_openai = OpenAI(api_key=openai_api_key, organization=openai_organization)


def get_response_openai(model, prompt, temperature=0.0, top_p=1.0, max_tokens=32):
    completion = client_openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    response = completion.choices[0].message.content
    return response


prompt = """Below is a response from an AI model to a font recognition task:
<response>
{response}
</response>
Now, you need to determine the final answer provided by the AI model in the response and return it. If there is no final or decided answer in the response, return 'None' or 'Undecided'."""


folder_path = "./results"
for root, _, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(root, file).replace("\\", "/")
            new_file_path = "./results_processed/" + file_path.split("/")[2] + "/" + file_path.split("/")[3] + "/" + file_path.split("/")[4]
            if os.path.exists(new_file_path):
                continue
            else:
                print(f"Processing: {file_path}")
                processed_data = []
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for data_piece in tqdm(data, total=len(data)):
                        img_path = data_piece["img_path"]
                        label = data_piece["label"]
                        response = data_piece["response"]
                        judge = data_piece["judge"]
                        final_response = get_response_openai(model="gpt-4o-mini", prompt=prompt.format(response=response))
                        if final_response.lower() in label.lower() or label.lower() in final_response.lower():
                            final_judge = True
                        else:
                            final_judge = False
                        processed_data.append({"img_path": img_path, "label": label, "response": response, "final_response": final_response, "judge": judge, "final_judge": final_judge})
                        os.makedirs("./results_processed/" + file_path.split("/")[2] + "/" + file_path.split("/")[3], exist_ok=True)
                        new_file_path = "./results_processed/" + file_path.split("/")[2] + "/" + file_path.split("/")[3] + "/" + file_path.split("/")[4]
                        with open(new_file_path, "w", encoding="utf-8") as f:
                            json.dump(processed_data, f, ensure_ascii=False, indent=4)
