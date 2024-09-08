import tiktoken

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4o-mini")

assert enc.decode(enc.encode("hello world")) == "hello world"

print(enc.encode("Yes"))

print(enc.encode("No"))

print(enc.encode("<|end|>"))