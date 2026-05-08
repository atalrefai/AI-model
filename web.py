import gradio as gr
from llama_cpp import Llama

class PhiChatbot:
    def __init__(self, model_path):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,
            verbose=False
        )
    
    def respond(self, message, history):
        """يبني الرد مع الاحتفاظ بالسياق الكامل للمحادثة"""
        # تحويل تاريخ المحادثة إلى تنسيق messages
        messages = []
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": message})
        
        # استدعاء النموذج
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            stop=["<|end|>", "<|user|>", "<|assistant|>"],
            temperature=0.7
        )
        
        bot_reply = response['choices'][0]['message']['content'].strip()
        return bot_reply

# إنشاء كائن الشات بوت
chatbot = PhiChatbot("Phi-3-mini-4k-instruct-q4.gguf")

# بناء واجهة Gradio
demo = gr.ChatInterface(
    fn=chatbot.respond,
    title="مساعدي الذكي المحلي - Phi-3-Mini",
    description="نموذج يعمل بالكامل على جهازك ويفهم سياق المحادثة"
)

if __name__ == "__main__":
    demo.launch()