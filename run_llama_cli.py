from llama_cpp import Llama

# تحميل النموذج
llm = Llama(
    model_path="Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=2048,         # حجم نافذة السياق
    n_threads=8         # عدد أنوية المعالج
)

# اختبار المحادثة
output = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "مرحباً، كيف يمكنني تعلم لغة بايثون؟"}
    ],
    max_tokens=256,
    stop=["<|end|>"]
)

print(output['choices'][0]['message']['content'])