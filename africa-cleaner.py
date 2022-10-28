import random
import string
import re

charset_file_path = 'data/africa-charset.txt'

charset_string = open(charset_file_path, encoding='utf-8').read()

unk_char = "#"  # unknown word to replace with unknown tokens
unk_word = "<###>"

def clean(line):
    
    line = line.strip()  

    # replace unknown characters by '#'
    line = re.sub(
        f'[^{charset_string} ]', unk_char, line)
    line = re.sub(f'(\#+\/?)+', f' {unk_word} ', line)

    line = re.sub(f'\s({unk_word}\s*)+', f' {unk_word} ', line)
    line = re.sub(f'(\s({unk_word}\s)[!"_\#\$\%\&\'\(\)\*\+,\-\.\/\:\;<=>\?\@[\\]\^`{{|}}~]+)+', f' {unk_word} ', line)
    line = re.sub(f'\s({unk_word}\s*)+', f' {unk_word} ', line)
    
    line = re.sub(r'\s+', " ", line).strip()
    line = line.strip()
    return line


def clean_lines(lines):
    lines = [clean(line.strip()).strip() for line in lines.split('\n')]
    return " ".join(lines)


def clean_file(in_path, out_path, keep_line_prob=1.0, min_token_per_line=1):
    with open(in_path, encoding='utf-8') as fin:
        with open(out_path, encoding='utf-8', mode='w') as fout:
            for line in fin:

                if random.random() < keep_line_prob:
                    line = clean(line)

                    line = line.strip()
                    is_not_valid = bool(re.match('[!"#$%&\'()*+,-\\.\/:;<=>?@\[\]^_`{|}~፠፡።፣፤፥፦፧፨፩፪፫፬፭፮፯፰፱፲፳፴፵፶፷፸፹፺፻፼\d ]+$', line))

                    if not is_not_valid and len(line.split()) >= min_token_per_line:
                        fout.write(line + "\n")


clean_file('dataset/corpus/africa/africa-train.txt', 'dataset/corpus/africa/clean-africa-corpus-25.txt', keep_line_prob=0.25)
clean_file('dataset/corpus/africa/africa-train.txt', 'dataset/corpus/africa/clean-africa-corpus-50.txt', keep_line_prob=0.50)
clean_file('dataset/corpus/africa/africa-train.txt', 'dataset/corpus/africa/clean-africa-corpus-75.txt', keep_line_prob=0.75)