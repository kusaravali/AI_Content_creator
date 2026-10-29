"""
AI Content Creator
Generates social media content using the open-source
HuggingFaceTB/SmolLM2-360M-Instruct model (free, runs locally).
"""

import torch
from transformers import pipeline

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
OUTPUT_FILE = "student_ai_content.txt"


def load_model():
    """Load the text-generation model (downloads on first run)."""
    device = 0 if torch.cuda.is_available() else -1
    print("Loading model... (first run downloads ~700 MB)")
    print("Device:", "GPU" if device == 0 else "CPU")
    generator = pipeline("text-generation", model=MODEL_NAME, device=device)
    print("Model loaded successfully!\n")
    return generator


def create_content(generator, topic, platform, tone, language):
    """Generate a full content package for the given inputs."""
    prompt = (
        f"You are a social media content creator.\n"
        f"Create content in {language}.\n"
        f"Topic: {topic}\n"
        f"Platform: {platform}\n"
        f"Tone: {tone}\n"
        f"Give me:\n"
        f"1. Catchy title\n"
        f"2. Strong hook\n"
        f"3. Short caption\n"
        f"4. 30-second video script\n"
        f"5. 5 hashtags\n"
        f"6. Call to action\n"
        f"Keep it clear, useful and engaging."
    )

    result = generator(
        prompt,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        return_full_text=False,
    )
    return result[0]["generated_text"]


def generate_ideas(generator, topic):
    """Generate 5 different post ideas for a topic."""
    prompt = f"Create 5 different social media post ideas about {topic}."
    result = generator(
        prompt,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.8,
        return_full_text=False,
    )
    return result[0]["generated_text"]


def save_to_file(content, filename=OUTPUT_FILE):
    """Save generated content to a text file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nSaved as {filename}")


def main():
    print("=" * 30)
    print("AI CONTENT CREATOR")
    print("=" * 30)

    generator = load_model()

    topic = input("Enter topic (e.g. Machine Learning): ")
    platform = input("Enter platform (e.g. Instagram): ")
    tone = input("Enter tone (e.g. Friendly): ")
    language = input("Enter language (e.g. English/Hindi/Telugu): ")

    print("\nGenerating content... please wait.")
    content = create_content(generator, topic, platform, tone, language)

    print("\n===== YOUR AI CONTENT =====\n")
    print(content)
    save_to_file(content)

    more = input("\nGenerate 5 extra post ideas too? (y/n): ").strip().lower()
    if more == "y":
        ideas = generate_ideas(generator, topic)
        print("\n===== POST IDEAS =====\n")
        print(ideas)
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write("\n\n===== POST IDEAS =====\n\n" + ideas)
        print(f"\nIdeas added to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
