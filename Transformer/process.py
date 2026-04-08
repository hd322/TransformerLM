import os
import numpy as np
from tqdm import tqdm
from basics.tokenizer import Tokenizer  # 导入你的分词器

def preprocess_txt_to_npy_custom(
    input_txt_path: str,
    output_npy_path: str,
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens: list[str] = None
):
    print("1. Loading custom tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath,
        special_tokens=special_tokens
    )

    print(f"2. Reading and encoding {input_txt_path} ...")
    
    # 构建一个按行读取的文件生成器，极大节省内存
    def file_iterator(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                yield line

    # 使用你写的 encode_iterable 方法
    token_iterator = tokenizer.encode_iterable(file_iterator(input_txt_path))

    # 将生成的 token ID 收集成列表
    # 如果文件有几 GB 那么大，收集 list 依然会占一定内存，
    # 但由于只是存 int，比存整个字符串对象要好得多。
    tokens = list(tqdm(token_iterator, desc="Encoding text"))

    print("3. Converting to numpy array...")
    
    # 检查最大 ID 以防溢出。如果 vocab size > 65535，需要改用 np.int32
    max_id = max(tokenizer.vocab.keys())
    dtype = np.uint16 if max_id <= 65535 else np.int32
    
    token_array = np.array(tokens, dtype=dtype)

    print(f"Total tokens: {len(token_array):,}")

    # 4. 保存为 npy
    np.save(output_npy_path, token_array)
    print(f"Successfully saved to {output_npy_path}")

if __name__ == "__main__":
    # 配置你的文件路径
    INPUT_TXT = "./TinyStoriesV2-GPT4-train.txt"
    OUTPUT_NPY = "data_train.npy"
    VOCAB_FILE = "vocab.json"   # 替换为你的 vocab 路径
    MERGES_FILE = "merges.txt"  # 替换为你的 merges 路径
    
    # 你的特殊 token，例如 GPT-2 的 <|endoftext|>
    SPECIAL_TOKENS = ["<|endoftext|>"] 
    
    preprocess_txt_to_npy_custom(
        input_txt_path=INPUT_TXT,
        output_npy_path=OUTPUT_NPY,
        vocab_filepath=VOCAB_FILE,
        merges_filepath=MERGES_FILE,
        special_tokens=SPECIAL_TOKENS
    )