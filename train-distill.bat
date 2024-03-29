set lang=tig

@REM python train-distill.py --accelerator "gpu" --max_epochs 128 --fasttext-path "dataset/corpus/%lang%/%lang%-ft.vec" --word2vec-path "dataset/corpus/%lang%/%lang%-w2v.vec" --corpus "dataset/corpus/%lang%/clean-%lang%-corpus.txt" --charset-path "data/am-charset.txt" --vector-load-ratio 1.0 --train-ratio .99 --exp-name "new-tig-distill-%lang%-large" --early-stop 0 --step_gamma 0.94 --neg_seq_len 32 --model-size "large"

@REM python train_distill.py --accelerator "gpu" --max_epochs 64 --fasttext-path "dataset/corpus/%lang%/%lang%-ft.vec" --word2vec-path "dataset/corpus/%lang%/%lang%-w2v.vec" --corpus "dataset/corpus/%lang%/clean-%lang%-corpus.txt" --charset-path "data/am-charset.txt" --vector-load-ratio 1.0 --train-ratio .99 --exp-name "%lang%-small-w2v" --early-stop 0 --step_gamma 0.94 --neg_seq_len 32 --model-size "small"

set lang=am

python train_distill.py --accelerator "gpu" --max_epochs 128 --fasttext-path "dataset/corpus/%lang%/new-vecs/%lang%-ft-new.vec" --word2vec-path "dataset/corpus/%lang%/new-vecs/%lang%-w2v-new.vec" --corpus "dataset/corpus/%lang%/x-clean-%lang%-corpus.txt" --charset-path "data/am-charset.txt" --vector-load-ratio 1. --train-ratio 1. --exp-name "distill-%lang%-SM-BD" --early-stop 0 --step_gamma 0.94 --neg_seq_len 32 --model-size "small"

@REM python train-distill.py --accelerator "gpu" --max_epochs 128 --fasttext-path "dataset/corpus/%lang%/%lang%-ft-w1.vec" --word2vec-path "dataset/corpus/%lang%/%lang%-w2v.vec" --corpus "dataset/corpus/%lang%/clean-%lang%-corpus.txt" --charset-path "data/am-charset.txt" --vector-load-ratio 0.55 --train-ratio .99 --exp-name "new-tig-distill-%lang%-small" --early-stop 0 --step_gamma 0.94 --neg_seq_len 32 --model-size "small"
@REM set lang=africa

@REM @REM python train-distill.py --accelerator "gpu" --max_epochs 128 --fasttext-path "dataset/corpus/%lang%/%lang%-ft.vec" --word2vec-path "dataset/corpus/%lang%/%lang%-w2v.vec" --corpus "dataset/corpus/%lang%/clean-%lang%-corpus.txt" --charset-path "data/am-charset.txt" --vector-load-ratio 1.0 --train-ratio .99 --exp-name "new-tig-distill-%lang%-large" --early-stop 0 --step_gamma 0.94 --neg_seq_len 32 --model-size "large"

@REM python train_distill.py --accelerator "gpu" --max_epochs 64 --fasttext-path "dataset/corpus/%lang%/%lang%-ft.vec" --word2vec-path "dataset/corpus/%lang%/%lang%-w2v.vec" --corpus "dataset/corpus/%lang%/clean-%lang%-corpus.txt" --charset-path "data/africa-charset.txt" --vector-load-ratio 0.7 --train-ratio .99 --exp-name "%lang%-small-w2v" --early-stop 0 --step_gamma 0.94 --neg_seq_len 32 --model-size "small"
