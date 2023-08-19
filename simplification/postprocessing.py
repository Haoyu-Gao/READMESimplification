from nltk.tokenize import SpaceTokenizer


special_tokens = set(["<sos>", "<eos>", "<UNK>", "<code_small>",
                      "<url>", "<table>", "<line_sep>", "<requirement>",
                      "<code_large>", "<pad>"])


def postprocessing(file_path):
    tokenizer = SpaceTokenizer()

    new_lines = []
    with open(file_path) as f:
        for line in f.readlines():
            tokens = tokenizer.tokenize(line)
            cache = ""
            new_line = ""
            for i in range(len(tokens)):
                if tokens[i] == "<eos>":
                    break
                elif tokens[i].startswith("##"):
                    cache += tokens[i].strip("##")
                else:
                    new_line += (" " + cache)
                    cache = tokens[i]
            new_line += (" " + cache)
            new_line = new_line.strip()
            new_line = new_line.lstrip("<code_small> ")
            new_lines.append(new_line)

    with open(file_path, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')


if __name__ == '__main__':
    postprocessing("../md_simplification_data/transfer36.gen")
