import os
from tqdm import tqdm
from generator import text_to_image

font_names = [name.split(".")[0] for name in os.listdir("./fonts")]

alphabet = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

sentences = [
    "Yesterday, Max drove quite far to grab a zesty pizza slice, fresh juice, a weird box of quirky vitamins, and then headed home.",
    "Sylvia quickly typed her new quiz on a fancy laptop, while Jack fixed a broken gaming mouse and drank zebra-striped coffee.",
    "My neighbor Jade swiftly zipped across town in a quirky taxi to deliver fresh donuts, baked pizza, and some extra juice.",
    "Quincy faxed a wacky memo to his boss, just highlighting zero budget for extra zebra-themed pencils this week.",
    "Lucas quickly arranged a picnic with Zoe, bringing fresh fruits, jazz music, extra napkins, and a warm velvet blanket to enjoy.",
    "Priscilla drove to the zoo with Victor, examining quirky zebras, foxes, and a jovial monkey who loved peanuts.",
    "Gabriel watched an exciting quiz show, then drank fizzy soda while munching on a quick pizza slice for supper, after a movie.",
    "Felix quickly discovered a hidden note behind the wardrobe, just suggesting bizarre facts about quantum physics.",
    "Brianna typed her final exam in a cozy library nook, quietly watching a zebra crossing video and just sipping fizzy lemonade.",
    "Charlie quickly baked twenty extra muffins for Zoe's yard sale, hoping to amaze hungry folks with very jazzy toppings."
]


print("Generating Images With Font Names")
print("-" * 30)
for idx, true_label in tqdm(enumerate(font_names), total=len(font_names)):
    os.makedirs(f"./data/name/{true_label}", exist_ok=True)
    for true_font_content in font_names:
        text_to_image(
            text=true_font_content,
            font_path=os.path.join(os.getcwd(), f"fonts/{true_label}.ttf"),
            font_size=40,
            output_path=f"./data/name/{true_label}/{true_font_content}.jpg",
            max_width=None
        )
print("-" * 30)
print("Finish Generating Images With Font Names")


print("Generating Images With Sentences")
print("-" * 30)
for idx, true_label in tqdm(enumerate(font_names), total=len(font_names)):
    os.makedirs(f"./data/sentence/{true_label}", exist_ok=True)
    for sentence in sentences:
        text_to_image(
            text=sentence,
            font_path=os.path.join(os.getcwd(), f"fonts/{true_label}.ttf"),
            font_size=32,
            output_path=f"./data/sentence/{true_label}/{sentence.split(' ')[0]}.jpg",
            max_width=500
        )
print("-" * 30)
print("Finish Generating Images With Sentences")


print("Generating Images With Single Character in the Alphabet")
print("-" * 30)
for idx, true_label in tqdm(enumerate(font_names), total=len(font_names)):
    os.makedirs(f"./data/single/{true_label}", exist_ok=True)
    for true_font_content in alphabet:
        text_to_image(
            text=true_font_content,
            font_path=os.path.join(os.getcwd(), f"fonts/{true_label}.ttf"),
            font_size=80,
            output_path=f"./data/single/{true_label}/{true_font_content}.jpg",
            max_width=None
        )
print("-" * 30)
print("Finish Generating Images With Single Character in the Alphabet")
