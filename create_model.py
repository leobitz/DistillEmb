
from distill_emb_large import DistillEmbLarge
from distill_emb_mid import DistillEmbMid
from distill_emb_small import DistillEmbSmall


def create_distill_emb(char2int, dropout=0.0, output_size=300, pad_char=' ', model_size='large'):
    if model_size == "small":
        class_name = DistillEmbSmall
    elif model_size == "mid":
        class_name = DistillEmbMid
    elif model_size == "large":
        class_name = DistillEmbLarge

    model = class_name(char2int,  
                        output_size=output_size, 
                        pad_char=pad_char, 
                        dropout=dropout)
    return model
