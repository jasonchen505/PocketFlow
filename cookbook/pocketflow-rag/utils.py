import os
import numpy as np
from openai import OpenAI
# import nltk

def call_llm(prompt):    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""),base_url ="https://open.bigmodel.cn/api/paas/v4/")
    r = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content


def get_embedding(text):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""),base_url ="https://open.bigmodel.cn/api/paas/v4/")
    
    response = client.embeddings.create(
        model="embedding-3",
        input=text
    )
    
    # Extract the embedding vector from the response
    embedding = response.data[0].embedding
    
    # Convert to numpy array for consistency with other embedding functions
    return np.array(embedding, dtype=np.float32)

def fixed_size_chunk(text, chunk_size=2000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    return chunks

# def sentence_based_chunk(text, max_sentences=3):
#     sentences = nltk.sent_tokenize(text)
#     chunks = []
#     for i in range(0, len(sentences), max_sentences):
#         chunks.append(" ".join(sentences[i : i + max_sentences]))
#     return chunks

if __name__ == "__main__":
    print("=== Testing call_llm ===")
    # prompt_test1 = "南泉斩猫,意旨如何"
    # # print(call_llm(prompt_test1))
    # prompt = "In a few words, what is the meaning of life?"
    # print(f"Prompt: {prompt}")
    # response = call_llm(prompt)
    # print(f"Response: {response}")
    # print("=== Testing embedding function ===")  
    # text1 = "如击石火"
    # text2 = "似闪电光"
    # oai_emb1 = get_embedding(text1)
    # oai_emb2 = get_embedding(text2)
    # print(f"OpenAI Embedding 1 shape: {oai_emb1.shape}")
    # oai_similarity = np.dot(oai_emb1, oai_emb2)
    # print(f"OpenAI similarity between texts: {oai_similarity:.4f}")

    # print(10000**(1/256))