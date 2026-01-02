from datasets import load_dataset

def load_wikitext(split="train"):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n".join(dataset["text"])
    return text

if __name__ == "__main__":
    text = load_wikitext("train")
    print("Chars:", len(text))
    print(text[:500])
