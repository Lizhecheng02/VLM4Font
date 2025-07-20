import base64
import requests
import torch
import json
import os
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq, MllamaForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
from openai import OpenAI


def get_font_choices(root_path):
    if not os.path.exists(root_path):
        raise ValueError(f"Path Does Not Exist: {root_path}")

    subfolders = []
    for dirpath, _, _ in os.walk(root_path):
        if dirpath != root_path:
            rel_path = os.path.relpath(dirpath, root_path)
            subfolders.append(rel_path)

    return ", ".join(subfolders)


def get_zero_shot_prompt():
    return "What font is used in the text of the image?"


def get_zero_shot_CoT_prompt():
    return "What font is used in the text of the image? You need to think step by step."


def get_mcq_prompt():
    choices = get_font_choices(root_path=f"./data/name")
    prompt = f"What font is used in the text of the image? You must choose one specific font name from the following list: [{choices}]."
    return prompt


def get_mcq_CoT_prompt():
    choices = get_font_choices(root_path=f"./data/name")
    prompt = f"What font is used in the text of the image? You must choose one specific font name from the following list: [{choices}]. You need to think step by step."
    return prompt


def get_examples(num_shots, image_path, type, prompt):
    search_json_path = f"./search_results/search_top-{num_shots}.json"
    with open(search_json_path, "r") as file:
        similar_dict = json.load(file)

    image_path = image_path.replace("\\", "/")
    if similar_dict[image_path]:
        similar_image_paths = similar_dict[image_path]
    else:
        print(f"Image Path Not Found: {image_path}")

    examples = []
    for similar_image_path in similar_image_paths:
        if type == "fs":
            example = {"image_path": similar_image_path, "prompt": prompt, "response": f"The font used for the text in the image is {similar_image_path.split('/')[-2]}"}
            examples.append(example)
        elif type == "mcqfs":
            example = {"image_path": similar_image_path, "prompt": prompt, "response": f"The font used for the text in the image is {similar_image_path.split('/')[-2]}"}
            examples.append(example)

    return examples


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_response_vlm_openai(model, prompt, image_path, openai_api_key, temperature=0.0, top_p=0.95, max_tokens=512):
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]


def get_response_vlm_openai_few_shot(model, prompt, image_path, openai_api_key, examples=None, temperature=0.0, top_p=0.95, max_tokens=512):
    messages = []
    if examples:
        for example in examples:
            example_base64 = encode_image(example["image_path"])
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{example_base64}"}
                    },
                    {
                        "type": "text",
                        "text": example["prompt"]
                    }
                ]
            })
            messages.append({
                "role": "assistant",
                "content": example["response"]
            })

    base64_image = encode_image(image_path)
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    })

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]


def get_response_vlm_gemini(model, prompt, image_path, openrouter_api_key, temperature=0.0, top_p=0.95, max_tokens=512):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key
    )

    base64_image = encode_image(image_path)
    completion = client.chat.completions.create(
        extra_body={},
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content


def get_response_vlm_gemini_few_shot(model, prompt, image_path, openrouter_api_key, examples=None, temperature=0.0, top_p=0.95, max_tokens=512):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key
    )

    messages = []
    if examples:
        for example in examples:
            example_base64 = encode_image(example["image_path"])
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{example_base64}"}
                    },
                    {
                        "type": "text",
                        "text": example["prompt"]
                    }
                ]
            })
            messages.append({
                "role": "assistant",
                "content": example["response"]
            })

    base64_image = encode_image(image_path)
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
    })

    completion = client.chat.completions.create(
        extra_body={},
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content


def get_response_vlm_claude(model, prompt, image_path, openrouter_api_key, temperature=0.0, top_p=0.95, max_tokens=512):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key
    )

    base64_image = encode_image(image_path)
    completion = client.chat.completions.create(
        extra_body={},
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content


def get_response_vlm_qwen72(model, prompt, image_path, openrouter_api_key, temperature=0.0, top_p=0.95, max_tokens=512):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key
    )

    base64_image = encode_image(image_path)
    completion = client.chat.completions.create(
        extra_body={},
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content


def get_response_vlm_qwen72_few_shot(model, prompt, image_path, openrouter_api_key, examples=None, temperature=0.0, top_p=0.95, max_tokens=512):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key
    )

    messages = []
    if examples:
        for example in examples:
            example_base64 = encode_image(example["image_path"])
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{example_base64}"}
                    },
                    {
                        "type": "text",
                        "text": example["prompt"]
                    }
                ]
            })
            messages.append({
                "role": "assistant",
                "content": example["response"]
            })

    base64_image = encode_image(image_path)
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
    })

    completion = client.chat.completions.create(
        extra_body={},
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content


def get_response_vlm_llama90(model, prompt, image_path, openrouter_api_key, temperature=0.0, top_p=0.95, max_tokens=512):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key
    )

    base64_image = encode_image(image_path)
    completion = client.chat.completions.create(
        extra_body={},
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content


def get_response_vlm_llama90_few_shot(model, prompt, image_path, openrouter_api_key, examples=None, temperature=0.0, top_p=0.95, max_tokens=512):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key
    )

    messages = []
    if examples:
        for example in examples:
            example_base64 = encode_image(example["image_path"])
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{example_base64}"}
                    },
                    {
                        "type": "text",
                        "text": example["prompt"]
                    }
                ]
            })
            messages.append({
                "role": "assistant",
                "content": example["response"]
            })

    base64_image = encode_image(image_path)
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
    })

    completion = client.chat.completions.create(
        extra_body={},
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content


def load_qwen2vl_model(model_name):
    if "qwen2.5" in model_name.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def qwen2_vlm_inference(model, processor, prompt, image_path, temperature=0.0, top_p=0.95, max_new_tokens=512):
    base64_image = encode_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image;base64,{base64_image}"
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output_text


def qwen2_vlm_inference_few_shot(model, processor, prompt, image_path, examples=None, temperature=0.0, top_p=0.95, max_new_tokens=512):
    messages = []
    if examples:
        for example in examples:
            example_base64 = encode_image(example["image_path"])
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image;base64,{example_base64}"
                    },
                    {"type": "text", "text": example["prompt"]}
                ]
            })
            messages.append({
                "role": "assistant",
                "content": example["response"]
            })

    base64_image = encode_image(image_path)
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": f"data:image;base64,{base64_image}"
            },
            {"type": "text", "text": prompt}
        ]
    })

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output_text


def load_phi_vision_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        num_crops=16
    )
    return model, processor


def phi_vlm_inference(model, processor, prompt, image_path, temperature=0.0, top_p=0.95, max_new_tokens=512):
    images = []
    placeholder = ""
    images.append(Image.open(image_path))
    placeholder += f"<|image_1|>\n"

    messages = [{"role": "user", "content": placeholder + prompt}]
    input = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(input, images, return_tensors="pt").to(model.device)

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "temperature": temperature
    }

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args, do_sample=False)

    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response


def phi_vlm_inference_few_shot(model, processor, prompt, image_path, examples=None, temperature=0.0, top_p=0.95, max_new_tokens=512):
    images = []
    messages = []
    if examples:
        for idx, example in enumerate(examples, start=1):
            images.append(Image.open(example["image_path"]))
            example_content = f"<|image_{idx}|>\n{example['prompt']}"
            messages.append({"role": "user", "content": example_content})
            messages.append({"role": "assistant", "content": example["response"]})

    images.append(Image.open(image_path))
    target_content = f"<|image_{len(images)}|>\n{prompt}"
    messages.append({"role": "user", "content": target_content})

    input = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(input, images, return_tensors="pt").to(model.device)

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "temperature": temperature
    }

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args, do_sample=False)

    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response


def load_idefics_model(model_name):
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def idefics_vlm_inference(model, processor, prompt, image_path, temperature=0.0, top_p=0.95, max_new_tokens=512):
    image = Image.open(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    input = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=input, images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, do_sample=False)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].split("\nAssistant:")[-1].strip()

    return generated_text


def idefics_vlm_inference_few_shot(model, processor, prompt, image_path, examples=None, temperature=0.0, top_p=0.95, max_new_tokens=512):
    messages = []
    images = []
    if examples:
        for example in examples:
            example_image = Image.open(example["image_path"])
            images.append(example_image)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["prompt"]}
                ]
            })
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": example["response"]}
                ]
            })

    target_image = Image.open(image_path)
    images.append(target_image)
    messages.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]
    })

    input = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=input, images=images, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, do_sample=False)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].split("\nAssistant:")[-1].strip()

    return generated_text


def load_llama_vision_model(model_name, hf_token):
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token
    )
    processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
    return model, processor


def llama32_vlm_inference(model, processor, prompt, image_path, temperature=0.0, top_p=0.95, max_new_tokens=512):
    image = Image.open(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, do_sample=False)
    generated_text = processor.decode(output[0]).split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0].strip()
    return generated_text


def llama32_vlm_inference_few_shot(model, processor, prompt, image_path, examples=None, temperature=0.0, top_p=0.95, max_new_tokens=512):
    messages = []
    images = []
    if examples:
        for example in examples:
            image = Image.open(example["image_path"])
            images.append(image)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["prompt"]}
                ]
            })
            messages.append({
                "role": "assistant",
                "content": example["response"]
            })

    image = Image.open(image_path)
    images.append(image)
    messages.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]
    })

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        images,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, do_sample=False)
    generated_text = processor.decode(output[0]).split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0].strip()
    return generated_text
