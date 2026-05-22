import tiktoken # For token counting

# --- Configuration ---
# For OpenAI models, 'cl100k_base' is common for gpt-3.5-turbo, gpt-4
ENCODING_NAME = "cl100k_base"
# Example hypothetical costs (USD per 1K tokens) for illustration.
# Actual costs vary by model and provider (e.g., OpenAI gpt-3.5-turbo-0125).
COST_PER_1K_TOKENS_INPUT = 0.0005  # Example cost for gpt-3.5-turbo input
COST_PER_1K_TOKENS_OUTPUT = 0.0015 # Example cost for gpt-3.5-turbo output

# --- Helper Function for Token Counting and Cost Estimation ---
def count_tokens_and_cost(text, is_input=True):
    """Counts tokens in a given text and estimates its cost."""
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    tokens = encoding.encode(text)
    num_tokens = len(tokens)
    
    if is_input:
        cost = (num_tokens / 1000) * COST_PER_1K_TOKENS_INPUT
    else:
        cost = (num_tokens / 1000) * COST_PER_1K_TOKENS_OUTPUT
        
    return num_tokens, cost

# --- Dummy Article Text (in Turkish, reflecting the article's language) ---
article_text = """
Büyük Dil Modelleri (LLM'ler), yapay zeka dünyasında devrim niteliğinde bir dönüşüm yaratırken, 
işletmeler için sundukları sınırsız potansiyelin yanı sıra ciddi maliyet endişelerini de beraberinde getiriyor. 
Özellikle API tabanlı LLM kullanımlarında, token başına ödeme modelleri ve artan kullanım hacimleri, 
faturaların hızla yükselmesine neden olabiliyor. Peki, bu güçlü teknolojiden faydalanmaya devam ederken, 
modelin sunduğu kaliteden taviz vermeden LLM maliyetlerini nasıl optimize edebiliriz? 
Bu makalede, bu sorunun cevabını adım adım, pratik stratejiler ve gerçek dünya senaryolarıyla keşfedeceğiz. 
Amacımız, hem başlangıç seviyesindeki kullanıcıların hem de deneyimli geliştiricilerin LLM maliyetlerini 
etkin bir şekilde yönetmelerini sağlayacak kapsamlı bir rehber sunmaktır.
"""

print("--- LLM Cost Optimization Example ---")
print(f"Using encoding: '{ENCODING_NAME}' for token counting")
print(f"Hypothetical costs: Input ${COST_PER_1K_TOKENS_INPUT}/1K tokens, Output ${COST_PER_1K_TOKENS_OUTPUT}/1K tokens\n")

# --- Scenario 1: Prompt Engineering (Optimizing Input Cost) ---
print("Scenario 1: Optimizing Input Prompt Length")

# Verbose Prompt for summarization
verbose_prompt = f"""
I need you to read the following article carefully. After reading it, please provide a comprehensive summary 
that captures all the main points and key takeaways. Make sure the summary is detailed enough for someone 
who hasn't read the original article to understand the core message. Here is the article:
{article_text}
"""
verbose_input_tokens, verbose_input_cost = count_tokens_and_cost(verbose_prompt, is_input=True)

print(f"  Verbose Prompt Input Tokens: {verbose_input_tokens}")
print(f"  Verbose Prompt Input Cost: ${verbose_input_cost:.6f}")

# Concise Prompt for the same summarization task
concise_prompt = f"""
Summarize the following article:
{article_text}
"""
concise_input_tokens, concise_input_cost = count_tokens_and_cost(concise_prompt, is_input=True)

print(f"  Concise Prompt Input Tokens: {concise_input_tokens}")
print(f"  Concise Prompt Input Cost: ${concise_input_cost:.6f}")

# --- Demonstrating the saving from prompt engineering ---
# This illustrates how a more concise prompt can reduce input tokens and cost,
# often without sacrificing the quality of the desired output.
print("\n  --- Comparison (Prompt Engineering) ---")
print(f"  Savings from concise prompt (input tokens): {verbose_input_tokens - concise_input_tokens} tokens")
print(f"  Savings from concise prompt (input cost): ${verbose_input_cost - concise_input_cost:.6f}")
print("  By making prompts more concise, we reduce input token count and thus input costs, often without impacting output quality significantly.")


# --- Scenario 2: Controlling Output Length (Optimizing Output Cost) ---
print("\nScenario 2: Controlling Output Length (using max_tokens parameter)")

# Simulate a long summary response (e.g., if max_tokens was set high)
long_summary_text = """
Büyük Dil Modelleri (LLM'ler), yapay zeka alanında önemli bir dönüşüm yaratmıştır. Ancak, API tabanlı kullanımlarda 
token bazlı ücretlendirme ve artan kullanım hacimleri nedeniyle ciddi maliyet endişeleri ortaya çıkmaktadır. 
Bu makale, LLM maliyetlerini kaliteden ödün vermeden nasıl optimize edeceğimizi pratik stratejilerle açıklamaktadır. 
Amaç, hem yeni başlayanların hem de deneyimli geliştiricilerin LLM maliyetlerini etkin bir şekilde yönetmelerine yardımcı olmaktır. 
LLM'ler metin oluşturma, özetleme, çeviri ve kod yazma gibi birçok görevi otomatize eder. 
Maliyetlerin temelinde token bazlı ücretlendirme yatar. Her kelime veya karakterin bir token olarak sayılması, 
özellikle uzun metinlerde veya yoğun kullanımlarda faturaların hızla yükselmesine neden olabilir. 
Bu nedenle, maliyetleri anlamak ve yönetmek, LLM teknolojisinden sürdürülebilir bir şekilde faydalanmak için kritik öneme sahiptir.
Bu stratejiler arasında prompt mühendisliği, model seçimi ve çıktı uzunluğunu sınırlama gibi yöntemler bulunmaktadır.
"""

# Simulate a short summary response (e.g., if max_tokens was set low)
short_summary_text = """
LLM'ler yapay zekada devrim yaratırken, API tabanlı kullanımlarda token bazlı maliyet endişeleri doğurur. 
Bu makale, kaliteden ödün vermeden LLM maliyetlerini optimize etme yollarını, özellikle prompt mühendisliği ve 
çıktı uzunluğu kontrolü gibi pratik stratejilerle ele almaktadır.
"""

long_output_tokens, long_output_cost = count_tokens_and_cost(long_summary_text, is_input=False)
short_output_tokens, short_output_cost = count_tokens_and_cost(short_summary_text, is_input=False)

print(f"  Long Summary Output Tokens (e.g., if max_tokens=200 was used): {long_output_tokens}")
print(f"  Long Summary Output Cost: ${long_output_cost:.6f}")

print(f"  Short Summary Output Tokens (e.g., if max_tokens=50 was used): {short_output_tokens}")
print(f"  Short Summary Output Cost: ${short_output_cost:.6f}")

# --- Demonstrating the saving from controlling output length ---
# This shows how limiting the LLM's output length (e.g., using the 'max_tokens' parameter
# in an API call) can significantly reduce output token count and associated costs.
print("\n  --- Comparison (Output Length Control) ---")
print(f"  Savings from controlling output length (output tokens): {long_output_tokens - short_output_tokens} tokens")
print(f"  Savings from controlling output length (output cost): ${long_output_cost - short_output_cost:.6f}")
print("  By setting appropriate 'max_tokens' for the LLM response, we can limit the output length and significantly reduce output costs, especially for tasks where conciseness is sufficient.")

print("\nThis example demonstrates how small changes in prompt engineering and output control can lead to tangible cost savings in LLM usage.")
