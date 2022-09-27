import random
import string
import re

charset_file_path = 'data/am-charset.txt'
token_replace_map_path = 'data/replace.txt'

charset_string = open(charset_file_path, encoding='utf-8').read()

unk_char = "#"  # unknown word to replace with unknown tokens
unk_word = "<###>"

replace_map = open(token_replace_map_path, encoding='utf-8').read().split('\n')
replace_map = {line.split('=')[0]: line.split('=')[1]
               for line in replace_map if len(line) > 0}

charset  = set(charset_string)

def clean(line):
    for ik, (k, v) in enumerate(replace_map.items()):
        line = line.replace(k, v)
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


clean_file('dataset/corpus/corpus.txt', 'dataset/corpus/clean-am-corpus.txt')