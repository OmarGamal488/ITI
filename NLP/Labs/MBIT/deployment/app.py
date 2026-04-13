"""
P2P: From Post To Personality — MBTI Prediction Demo
Based on Ma et al., CIKM 2025

Author: Omar Gamal ElKady | ITI - AI Track, Intake 46

Predicts Myers-Briggs personality type from social media posts using
DeepSeek-V3 API with the paper's exact prompt templates (Appendix C).
"""

import gradio as gr
import os
import re
from openai import OpenAI

# DeepSeek API client
api_key = os.environ.get("DEEPSEEK_API_KEY", "")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com") if api_key else None

# Paper Appendix C prompts
FEATURE_PROMPT = (
    "According to the following content <CONTENT>, extract the key features "
    "from these perspectives:\n"
    "1. Social tendency (Extraversion E / Introversion I)\n"
    "2. Information processing mode (Sensing S / iNtuition N)\n"
    "3. Decision-making mode (Thinking T / Feeling F)\n"
    "4. Lifestyle (Judging J / Perceiving P)"
)

PREDICT_PROMPT = (
    "According to the following content <CONTENT>, combined with the key features <FEATURES> "
    "extracted by the local model, "
    "predict the MBTI type from four dimensions (only output four letters). "
    "The emphases of the four dimensions are as follows:\n"
    "1. Social tendency (Extraversion E / Introversion I)\n"
    "2. Information processing mode (Sensing S / iNtuition N)\n"
    "3. Decision-making mode (Thinking T / Feeling F)\n"
    "4. Lifestyle (Judging J / Perceiving P)"
)

SYSTEM_PROMPT = (
    "You are an MBTI personality type classifier. "
    "You MUST respond with ONLY 4 uppercase letters representing the MBTI type. "
    "Valid types: INFP, INFJ, INTP, INTJ, ENFP, ENFJ, ENTP, ENTJ, "
    "ISFP, ISFJ, ISTP, ISTJ, ESFP, ESFJ, ESTP, ESTJ. "
    "No explanation, no other text. Just 4 letters."
)

VALID_TYPES = {
    "INFP", "INFJ", "INTP", "INTJ", "ENFP", "ENFJ", "ENTP", "ENTJ",
    "ISFP", "ISFJ", "ISTP", "ISTJ", "ESFP", "ESFJ", "ESTP", "ESTJ"
}

MBTI_INFO = {
    "INTJ": ("The Architect", "Imaginative and strategic thinkers with a plan for everything."),
    "INTP": ("The Logician", "Innovative inventors with an unquenchable thirst for knowledge."),
    "ENTJ": ("The Commander", "Bold, imaginative and strong-willed leaders."),
    "ENTP": ("The Debater", "Smart and curious thinkers who cannot resist an intellectual challenge."),
    "INFJ": ("The Advocate", "Quiet and mystical, yet very inspiring and tireless idealists."),
    "INFP": ("The Mediator", "Poetic, kind and altruistic people, always eager to help a good cause."),
    "ENFJ": ("The Protagonist", "Charismatic and inspiring leaders, able to mesmerize their listeners."),
    "ENFP": ("The Campaigner", "Enthusiastic, creative and sociable free spirits."),
    "ISTJ": ("The Logistician", "Practical and fact-minded individuals, whose reliability cannot be doubted."),
    "ISFJ": ("The Defender", "Very dedicated and warm protectors, always ready to defend their loved ones."),
    "ESTJ": ("The Executive", "Excellent administrators, unsurpassed at managing things or people."),
    "ESFJ": ("The Consul", "Extraordinarily caring, social and popular people."),
    "ISTP": ("The Virtuoso", "Bold and practical experimenters, masters of all kinds of tools."),
    "ISFP": ("The Adventurer", "Flexible and charming artists, always ready to explore and experience something new."),
    "ESTP": ("The Entrepreneur", "Smart, energetic and very perceptive people, who truly enjoy living on the edge."),
    "ESFP": ("The Entertainer", "Spontaneous, energetic and enthusiastic people, life is never boring around them."),
}

DIMENSION_NAMES = {
    "E": "Extraversion", "I": "Introversion",
    "S": "Sensing", "N": "Intuition",
    "T": "Thinking", "F": "Feeling",
    "J": "Judging", "P": "Perceiving",
}


def predict_mbti(posts: str) -> str:
    if not client:
        return "**Error:** DEEPSEEK_API_KEY not set. Add it as a Space secret in Settings."

    if not posts.strip() or len(posts.strip()) < 50:
        return "Please enter at least 50 characters of text (social media posts, writing samples, etc.)"

    try:
        # Step 1: Feature extraction (Paper Appendix C)
        feature_prompt = FEATURE_PROMPT.replace("<CONTENT>", posts[:4000])
        r1 = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": feature_prompt}],
            max_tokens=200,
            temperature=0,
        )
        features = r1.choices[0].message.content.strip()

        # Step 2: MBTI prediction (Paper Appendix C)
        predict_prompt = (PREDICT_PROMPT
            .replace("<CONTENT>", posts[:3000])
            .replace("<FEATURES>", features[:500]))

        r2 = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": predict_prompt}
            ],
            max_tokens=10,
            temperature=0,
        )
        raw = r2.choices[0].message.content.strip().upper()

        # Parse MBTI type
        if raw in VALID_TYPES:
            mbti = raw
        else:
            match = re.search(r"[EI][SN][TF][JP]", raw)
            mbti = match.group(0) if match else None

        if not mbti or mbti not in VALID_TYPES:
            return f"Could not parse a valid MBTI type from model response: {raw}"

        # Build result
        title, desc = MBTI_INFO.get(mbti, ("", ""))
        dims = []
        for i, letter in enumerate(mbti):
            label = DIMENSION_NAMES.get(letter, letter)
            dim_name = ["E/I", "S/N", "T/F", "J/P"][i]
            dims.append(f"- **{dim_name}:** {label} ({letter})")

        result = f"""## {mbti} — {title}

*{desc}*

### Dimensions
{chr(10).join(dims)}

### Personality Assessment
{features}

---
*Powered by DeepSeek-V3 | Fine-tuned model: [P2P-DeepSeek-R1-8B-MBTI-LoRA](https://huggingface.co/OmarGamal48812/P2P-DeepSeek-R1-8B-MBTI-LoRA)*
"""
        return result

    except Exception as e:
        return f"**Error:** {str(e)}"


# Examples
examples = [
    ["I love spending time alone reading books and thinking about abstract theories. "
     "I prefer deep one-on-one conversations over large group gatherings. "
     "I tend to make decisions based on logic rather than feelings. "
     "I like to keep my options open and often procrastinate on decisions. "
     "Philosophy and science fascinate me more than practical matters."],
    ["I'm the life of the party! I love meeting new people and trying new experiences. "
     "I go with my gut feeling when making decisions. "
     "I'm very organized and always have a plan for everything. "
     "I notice details that others miss and prefer practical solutions over theories. "
     "Helping others and maintaining harmony in my relationships is very important to me."],
    ["I spend a lot of time imagining different possibilities and future scenarios. "
     "I deeply care about other people's feelings and try to maintain harmony. "
     "I prefer a structured lifestyle with clear goals and deadlines. "
     "I get energized by spending time with close friends but need alone time to recharge. "
     "Art, music and creative expression are central to who I am."],
]

demo = gr.Interface(
    fn=predict_mbti,
    inputs=gr.Textbox(
        label="Paste your social media posts or writing samples",
        placeholder="Enter text here... (minimum 50 characters, more text gives better predictions)",
        lines=10,
    ),
    outputs=gr.Markdown(label="MBTI Prediction"),
    title="P2P: From Post To Personality",
    description=(
        "Predict your Myers-Briggs personality type from your writing.\n\n"
        "Based on: **Ma et al. (CIKM 2025)** — *From Post To Personality: "
        "Harnessing LLMs for MBTI Prediction in Social Media*\n\n"
        "**How it works:**\n"
        "1. DeepSeek-V3 extracts personality features from your text\n"
        "2. DeepSeek-V3 predicts your 4-letter MBTI type based on the features\n\n"
        "**Author:** Omar Gamal ElKady | ITI - AI Track, Intake 46"
    ),
    examples=examples,
    cache_examples=False,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
