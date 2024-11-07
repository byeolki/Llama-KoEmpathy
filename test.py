from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("outputs/checkpoint-1065", load_in_4bit=True)  

alpaca_prompt = """아래는 작업을 설명하는 지시사항입니다. 입력된 내용을 바탕으로 적절한 응답을 작성하세요.
### 지시사항:
입력에 대해서 공감해주세요.
### 입력:
{input}
### 응답:
"""

FastLanguageModel.for_inference(model) 
input_text = "나 살이 너무 많이 찐거 같아서 속상해."

inputs = tokenizer([alpaca_prompt.format(input=input_text)], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)

generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(generated_text)