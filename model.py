from huggingface_hub import InferenceClient
from token_file import HF_TOKEN
client = InferenceClient(token=HF_TOKEN)

def adv(user_inputs, prediction):
    prompt = f"Here are the student inputs: {user_inputs}. The predicted CGPA is {prediction:.2f}. Give advice to improve performance."

    response = client.text_generation(
        model="fdurant/colbert-xm-for-inference-api",
        prompt=prompt,
        max_new_tokens=200,
        temperature=0.7,
    )
    return response



if __name__ == "__main__":
    user_inputs = {
        "social": 5,
        "study_hours": 2,
        "sleep_hours": 3,
        "attendance": "86%",
        "depression": 4,
        "anxiety": 5,
        "stress": 6,
    }

    prediction = 62.5

    advice = adv(user_inputs, prediction)
    print("\n--- Counsellor’s Advice ---\n")
    print(advice)
