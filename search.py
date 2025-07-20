import os
import torch
import numpy as np
import random
import json
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_image_paths(data_dir):
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))
    return image_paths


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", size=224)
    return inputs


def extract_features(image_paths):
    features = []
    for image_path in tqdm(image_paths, desc="Extracting Features"):
        inputs = preprocess_image(image_path)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            features.append(image_features.squeeze(0).cpu().numpy())
    return np.array(features)


def find_similar_images(selected_image_path, database_image_paths, features, top_n=5):
    selected_inputs = preprocess_image(selected_image_path)
    with torch.no_grad():
        selected_feature = model.get_image_features(**selected_inputs).squeeze(0).cpu().numpy()

    similarities = cosine_similarity([selected_feature], features)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_paths = [database_image_paths[i] for i in sorted_indices]
    sorted_scores = [similarities[i] for i in sorted_indices]

    return sorted_paths[:top_n], sorted_scores[:top_n]


def search(database_image_paths, selected_image_path, features, top_n=5):
    similar_paths, _ = find_similar_images(selected_image_path, database_image_paths, features, top_n)
    return similar_paths


if __name__ == "__main__":
    image_paths = get_image_paths("./data/name") + get_image_paths("./data/sentence")
    search_image_paths = get_image_paths("./data/single")
    features = extract_features(image_paths)

    for top_n in [1, 2, 3, 4, 5, 6]:
        results = {}
        for selected_image in tqdm(image_paths, total=len(image_paths), desc="Searching"):
            category = selected_image.split("/")[3]
            similar_paths = search(database_image_paths=search_image_paths, selected_image_path=selected_image, features=features, top_n=top_n)
            random.shuffle(similar_paths)
            results[selected_image] = similar_paths

        os.makedirs("search_results", exist_ok=True)
        output_file = f"search_results/search_top-{top_n}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Save to {output_file}")
