import os
from openai import OpenAI
from datasets import Dataset, DatasetDict
from gte import GteModel
import numpy as np

client = OpenAI(organization='$ORG_ID',project='$PROJECT_ID',)
client.api_key = "$OPENAI_API_KEY"

def generate_questions(text):
  """Generates 10 questions from the given text using OpenAI."""
  p = '"Based on the following text extracted from a technical wiki, generate 10 questions that ordinary users could ask where answers can be found in the text. Output only the questions generated, do not add anything else. If the text has no meaningful content, do not attemp to generate any question, instead write "No content found". Text is: '+text+'"'
  prompt = [
    {"role":"system","content":"You are an expert in technical documentation. You are given a set of technical documentations and are tasked to think from the end user perspective what sort of questions they might raise where answers could be found within these documents."},
    {"role":"user", "content":p}
  ]
  response = client.chat.completions.create(
    model="gpt-4o-2024-05-13",
    messages=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
  )
  questions = response.choices[0].message.content.strip().split("\n")
  return questions

def generate_dataset(wiki_dir):
  """Generates a dataset for LORA training."""
  embed = GteModel()  # Replace with your GTE model loading logic
  data = {
    'id': [],
    'gte': [],
    'phi': []
  }
  id_counter = 1

  for file in os.listdir(wiki_dir):
    if file.startswith('.'):
        continue
    with open(os.path.join(wiki_dir, file), "r", encoding="utf-8", errors="ignore") as f:
      text = f.read()

    print(f"\nText:\n{text}\n")
    questions = generate_questions(text)
    for question in questions:
      print(f"Question: {question}")
      phi = f"{question}<|end|>{text}"
      gte_embedding = embed(phi)
      np_embedding = np.array(gte_embedding)
      data['id'].append(id_counter)
      data['gte'].append(np_embedding)
      data['phi'].append(phi)
      id_counter += 1

  dataset = Dataset.from_dict(data)
  return dataset

if __name__ == "__main__":
  wiki_dir = "wiki"
  dataset = generate_dataset(wiki_dir)
  dataset.save_to_disk("lora_dataset")
  # Export dataset to CSV
  dataset.to_csv("lora_dataset.csv")
