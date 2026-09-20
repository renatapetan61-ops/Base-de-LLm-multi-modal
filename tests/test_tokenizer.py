"""
Testes do Tokenizer do Apolo Zenith 1.9
"""

import unittest
from tokenizer.zenith_tokenizer import ZenithTokenizer

class TestZenithTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ZenithTokenizer()

    def test_special_tokens(self):
        self.assertEqual(self.tokenizer.pad_token_id, 0)
        self.assertEqual(self.tokenizer.bos_token_id, 1)
        self.assertEqual(self.tokenizer.eos_token_id, 2)

    def test_portuguese_and_code_roundtrip(self):
        text = "def criar_modelo():\n    # Teste de Português Brasileiro\n    return 'Apolo Zenith 1.9'"
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
        self.assertEqual(text, decoded)

    def test_special_token_encoding(self):
        text = "<|thought_start|> Analisando arquitetura <|thought_end|>"
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)
        self.assertEqual(text, decoded)

if __name__ == "__main__":
    unittest.main()
