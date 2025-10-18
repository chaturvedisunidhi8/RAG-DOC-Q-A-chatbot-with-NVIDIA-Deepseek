from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "your_api_key"  ##or load from environment variable
)

completion = client.chat.completions.create(
  model="deepseek-ai/deepseek-v3.1-terminus",
  messages=[{"role": "user", "content": "Hi! Please respond only in English. Introduce yourself briefly."}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=8192,
  extra_body={"chat_template_kwargs": {"thinking":True}},
  stream=True
)

for chunk in completion:
  reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
  if reasoning:
    print(reasoning, end="")
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")
  

