import os
import yaml
import json
import argparse
from utils import *
from collections import defaultdict
from tqdm import tqdm

config_path = os.path.join("config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

openai_api_key = config["openai"]["api_key"]
openai_organization = config["openai"]["organization"]
huggingface_token = config["huggingface"]["api_key"]
openrouter_api_key = config["openrouter"]["api_key"]


def parse_comma_separated_list(value):
    return value.split(",")


def run(args):
    models = args.models

    folder_images = defaultdict(list)
    for root, _, files in os.walk(f"./data/name"):
        images = [f for f in files if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png"]]
        if images:
            folder_name = os.path.basename(root)
            for img in images:
                img_path = os.path.join(root, img)
                folder_images[folder_name].append(img_path)

    for model_name in models:
        print("Running model:", model_name)
        print("-" * 60)
        if "phi" in model_name.lower():
            model, processor = load_phi_vision_model(model_name)
        elif "qwen" in model_name.lower() and "72" not in model_name.lower():
            model, processor = load_qwen2vl_model(model_name)
        elif "llama" in model_name.lower() and "idefics" not in model_name.lower() and "90" not in model_name.lower():
            model, processor = load_llama_vision_model(model_name, huggingface_token)
        elif "idefics" in model_name.lower():
            model, processor = load_idefics_model(model_name)

        results = []
        total_count = 0
        correct_count = 0
        for folder, images in folder_images.items():
            print(f"Folder: {folder}")
            for img_path in tqdm(images, total=len(images)):
                if "gpt" in model_name.lower():
                    response = get_response_vlm_openai(model=model_name, prompt=get_zero_shot_prompt(), image_path=img_path, openai_api_key=openai_api_key)
                elif "gemini" in model_name.lower():
                    response = get_response_vlm_gemini(model=model_name, prompt=get_zero_shot_prompt(), image_path=img_path, openrouter_api_key=openrouter_api_key)
                elif "claude" in model_name.lower():
                    response = get_response_vlm_gemini(model=model_name, prompt=get_zero_shot_prompt(), image_path=img_path, openrouter_api_key=openrouter_api_key)
                elif "qwen" in model_name.lower() and "72" in model_name.lower():
                    response = get_response_vlm_qwen72(model=model_name, prompt=get_zero_shot_prompt(), image_path=img_path, openrouter_api_key=openrouter_api_key)
                elif "phi" in model_name.lower():
                    response = phi_vlm_inference(model=model, processor=processor, prompt=get_zero_shot_prompt(), image_path=img_path)
                elif "qwen" in model_name.lower():
                    response = qwen2_vlm_inference(model=model, processor=processor, prompt=get_zero_shot_prompt(), image_path=img_path)
                elif "llama" in model_name.lower() and "idefics" not in model_name.lower() and "90" not in model_name.lower():
                    response = llama32_vlm_inference(model=model, processor=processor, prompt=get_zero_shot_prompt(), image_path=img_path)
                elif "llama" in model_name.lower() and "idefics" not in model_name.lower() and "90" in model_name.lower():
                    response = get_response_vlm_llama90(model=model_name, prompt=get_zero_shot_prompt(), image_path=img_path, openrouter_api_key=openrouter_api_key)
                elif "idefics" in model_name.lower():
                    response = idefics_vlm_inference(model=model, processor=processor, prompt=get_zero_shot_prompt(), image_path=img_path)

                if response.lower() in folder.lower() or folder.lower() in response.lower():
                    correct_count += 1
                    total_count += 1
                    result_entry = {"img_path": img_path.replace("\\", "/"), "label": folder, "response": response, "judge": True}
                    results.append(result_entry)
                else:
                    total_count += 1
                    result_entry = {"img_path": img_path.replace("\\", "/"), "label": folder, "response": response, "judge": False}
                    results.append(result_entry)

        print("-" * 60)
        print(f"Zero-Shot | Model: {model_name.split('/')[-1]} | Accuracy: {correct_count / total_count * 100: .2f}%")

        os.makedirs(f"./results/name/{model_name.split('/')[-1]}", exist_ok=True)
        output_filename = f"./results/name/{model_name.split('/')[-1]}/zs.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results Saved to {output_filename}")


def run_sentence(args):
    models = args.models

    folder_images = defaultdict(list)
    for root, _, files in os.walk(f"./data/sentence"):
        images = [f for f in files if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png"]]
        if images:
            folder_name = os.path.basename(root)
            for img in images:
                img_path = os.path.join(root, img)
                folder_images[folder_name].append(img_path)

    for model_name in models:
        print("Running model:", model_name)
        print("-" * 60)
        if "phi" in model_name.lower():
            model, processor = load_phi_vision_model(model_name)
        elif "qwen" in model_name.lower() and "72" not in model_name.lower():
            model, processor = load_qwen2vl_model(model_name)
        elif "llama" in model_name.lower() and "idefics" not in model_name.lower() and "90" not in model_name.lower():
            model, processor = load_llama_vision_model(model_name, huggingface_token)
        elif "idefics" in model_name.lower():
            model, processor = load_idefics_model(model_name)

        results = []
        total_count = 0
        correct_count = 0
        for folder, images in folder_images.items():
            print(f"Folder: {folder}")
            for img_path in tqdm(images, total=len(images)):
                if "gpt" in model_name.lower():
                    response = get_response_vlm_openai(model=model_name, prompt=get_zero_shot_prompt(), image_path=img_path, openai_api_key=openai_api_key)
                elif "gemini" in model_name.lower():
                    response = get_response_vlm_gemini(model=model_name, prompt=get_zero_shot_prompt(), image_path=img_path, openrouter_api_key=openrouter_api_key)
                elif "claude" in model_name.lower():
                    response = get_response_vlm_gemini(model=model_name, prompt=get_zero_shot_prompt(), image_path=img_path, openrouter_api_key=openrouter_api_key)
                elif "qwen" in model_name.lower() and "72" in model_name.lower():
                    response = get_response_vlm_qwen72(model=model_name, prompt=get_zero_shot_prompt(), image_path=img_path, openrouter_api_key=openrouter_api_key)
                elif "phi" in model_name.lower():
                    response = phi_vlm_inference(model=model, processor=processor, prompt=get_zero_shot_prompt(), image_path=img_path)
                elif "qwen" in model_name.lower():
                    response = qwen2_vlm_inference(model=model, processor=processor, prompt=get_zero_shot_prompt(), image_path=img_path)
                elif "llama" in model_name.lower() and "idefics" not in model_name.lower() and "90" not in model_name.lower():
                    response = llama32_vlm_inference(model=model, processor=processor, prompt=get_zero_shot_prompt(), image_path=img_path)
                elif "llama" in model_name.lower() and "idefics" not in model_name.lower() and "90" in model_name.lower():
                    response = get_response_vlm_llama90(model=model_name, prompt=get_zero_shot_prompt(), image_path=img_path, openrouter_api_key=openrouter_api_key)
                elif "idefics" in model_name.lower():
                    response = idefics_vlm_inference(model=model, processor=processor, prompt=get_zero_shot_prompt(), image_path=img_path)

                if response.lower() in folder.lower() or folder.lower() in response.lower():
                    correct_count += 1
                    total_count += 1
                    result_entry = {"img_path": img_path.replace("\\", "/"), "label": folder, "response": response, "judge": True}
                    results.append(result_entry)
                else:
                    total_count += 1
                    result_entry = {"img_path": img_path.replace("\\", "/"), "label": folder, "response": response, "judge": False}
                    results.append(result_entry)

        print("-" * 60)
        print(f"Zero-Shot | Model: {model_name.split('/')[-1]} | Accuracy: {correct_count / total_count * 100: .2f}%")

        os.makedirs(f"./results/sentence/{model_name.split('/')[-1]}", exist_ok=True)
        output_filename = f"./results/sentence/{model_name.split('/')[-1]}/zs.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results Saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=parse_comma_separated_list, required=True)
    args = parser.parse_args()
    run(args)
    run_sentence(args)
