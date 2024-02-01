import deepl

auth_key = "77a9add8-3ad1-05ba-530f-ead6d9fb5ea8"  # Replace with your key
translator = deepl.Translator(auth_key)

result = translator.translate_text("Hello, world!", target_lang="FR")
print(1)
print(result.text)  # "Bonjour, le monde !"