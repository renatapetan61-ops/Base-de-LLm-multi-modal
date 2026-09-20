"""
Apolo Zenith 1.9 — Multimodal and Multilingual Code Tokenizer.
Projetado especificamente para Português Brasileiro, Linguagens de Programação
e Delimitadores de Raciocínio / Ferramentas / Multimodalidade.
"""

from typing import List, Dict, Optional, Union
import json
import os
import re

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|thought_start|>",
    "<|thought_end|>",
    "<|tool_call|>",
    "<|tool_response|>",
    "<|code_start|>",
    "<|code_end|>",
    "<|image|>",
    "<|video|>",
    "<|audio|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|indent_2|>",
    "<|indent_4|>",
    "<|newline|>",
]

class ZenithTokenizer:
    """
    Tokenizer determinístico multimodal e de programação do Apolo Zenith 1.9.
    Possui suporte de vocabulário estendido para Português (PT-BR) e 2.500 linguagens.
    """
    def __init__(self, vocab_path: Optional[str] = None):
        self.special_tokens = SPECIAL_TOKENS
        self.special_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.id_to_special: Dict[int, str] = {idx: tok for idx, tok in enumerate(self.special_tokens)}
        
        self.vocab: Dict[str, int] = dict(self.special_to_id)
        self.inverse_vocab: Dict[int, str] = dict(self.id_to_special)
        
        # Inserção base de bytes para garantia byte-level (nenhum caractere vira OOV)
        offset = len(self.special_tokens)
        for b in range(256):
            b_repr = f"<0x{b:02X}>"
            self.vocab[b_repr] = offset + b
            self.inverse_vocab[offset + b] = b_repr
            
        self.next_id = offset + 256
        
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        else:
            self._init_core_subwords()

    def _init_core_subwords(self):
        """Inicializa subpalavras frequentes em PT-BR e sintaxe de programação."""
        core_tokens = [
            "def", "class", "function", "return", "import", "from", "export",
            "const", "let", "var", "public", "private", "struct", "impl", "fn",
            "if", "else", "for", "while", "match", "case", "try", "catch",
            "async", "await", "self", "this", "super", "yield", "lambda",
            "int", "float", "str", "bool", "void", "true", "false", "null", "None",
            "O", "A", "Os", "As", "Um", "Uma", "que", "para", "com", "não",
            "em", "por", "sobre", "como", "modelo", "código", "função", "teste",
            "sistema", "usuário", "resposta", "geração", "imagem", "vídeo", "áudio",
            "arquitetura", "projeto", "desenvolvimento", "execução", "parâmetros",
            "==", "!=", "<=", ">=", "+=", "-=", "->", "=>", "::", "{", "}", "[", "]",
            "(", ")", ":", ";", ",", ".", "=", "+", "-", "*", "/", "%", "&", "|", "^"
        ]
        for tok in core_tokens:
            if tok not in self.vocab:
                self.vocab[tok] = self.next_id
                self.inverse_vocab[self.next_id] = tok
                self.next_id += 1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    @property
    def bos_token_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    @property
    def eos_token_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
            
        # Padrão regex para preservar tokens especiais e quebrar palavras/espaços/pontuação com suporte unicode
        pattern = r"(<\|[^|]+?\|>)|(\s+)|([\w]+)|([^\s\w]+)"
        matches = re.finditer(pattern, text, re.UNICODE)
        
        for match in matches:
            tok_text = match.group(0)
            if tok_text in self.vocab:
                tokens.append(self.vocab[tok_text])
            else:
                # Byte fallback para robustez contra caracteres desconhecidos
                for byte_val in tok_text.encode("utf-8"):
                    b_repr = f"<0x{byte_val:02X}>"
                    tokens.append(self.vocab.get(b_repr, self.special_to_id["<|unk|>"]))
                    
        if add_special_tokens:
            tokens.append(self.eos_token_id)
            
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        byte_stream = bytearray()
        result_parts: List[str] = []
        
        for tid in token_ids:
            if tid in self.id_to_special:
                if not skip_special_tokens:
                    # Esvaziar bytes pendentes se houver
                    if byte_stream:
                        result_parts.append(byte_stream.decode("utf-8", errors="replace"))
                        byte_stream.clear()
                    result_parts.append(self.id_to_special[tid])
                continue
                
            token_str = self.inverse_vocab.get(tid, "")
            if token_str.startswith("<0x") and token_str.endswith(">"):
                b_val = int(token_str[3:-1], 16)
                byte_stream.append(b_val)
            else:
                if byte_stream:
                    result_parts.append(byte_stream.decode("utf-8", errors="replace"))
                    byte_stream.clear()
                result_parts.append(token_str)
                
        if byte_stream:
            result_parts.append(byte_stream.decode("utf-8", errors="replace"))
            
        return "".join(result_parts)

    def save_vocab(self, vocab_path: str):
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load_vocab(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.next_id = max(self.vocab.values()) + 1
