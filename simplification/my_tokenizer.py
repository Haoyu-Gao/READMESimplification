from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


unk_token = "<UNK>"
special_tokens = ["<UNK>", "<code_small>", "<url>", "<table>", "<line_sep>", "<requirement>", "<code_large>",
                  "<sos>", "<eos>", "<pad>"]
"""
Keep more tokens <url> <short_code> etc. 

"""


def train_tokenizer():
    tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
    trainer = WordPieceTrainer(min_frequency=1, special_tokens=special_tokens, vocab_size=40000)

    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train(["../simplification_data/train.dst", "../simplification_data/train.src",
                     "../md_simplification_data/train.dst", "../md_simplification_data/train.src"], trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<sos> $A <eos>",
        special_tokens=[
            ("<sos>", tokenizer.token_to_id("<sos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>", length=500)
    tokenizer.enable_truncation(max_length=500)
    tokenizer.save(f"my_tokenizer.json")


# if __name__ == '__main__':
#     train_tokenizer()
