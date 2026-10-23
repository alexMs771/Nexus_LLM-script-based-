from training.tokenizer import train_tokenizer


def test_bpe_and_byte_tokenizers_round_trip_unseen_identifier():
    text = "CREATE TABLE unfamiliar_customer_2049 (e_mail_address VARCHAR(255));"
    for kind in ("byte", "byte_bpe"):
        tokenizer = train_tokenizer(["CREATE TABLE customers (email VARCHAR(255));"], kind)
        assert tokenizer.decode(tokenizer.encode(text)) == text
