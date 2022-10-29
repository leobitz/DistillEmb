@REM python train-distill.py --accelerator "gpu" --max_epochs 64 --fasttext-path "dataset/corpus/tig/tig-ft-w1.vec" --word2vec-path "dataset/corpus/tig/tig-w2v.vec" --corpus "dataset/corpus/tig/clean-tig-corpus.txt" --charset-path "data/am-charset.txt" --vector-load-ratio 1.0 --train-ratio 1.0 --exp-name "tig-distill-full32-both" --early-stop 0 --step_gamma 0.94 --neg_seq_len 32

@REM set lang=am


@REM python train-distill.py --accelerator "gpu" --max_epochs 64 --fasttext-path "dataset/corpus/%lang%/%lang%-ft.vec" --word2vec-path "dataset/corpus/%lang%/%lang%-w2v.vec" --corpus "dataset/corpus/%lang%/clean-%lang%-corpus.txt" --charset-path "data/am-charset.txt" --vector-load-ratio 0.5 --train-ratio 1.0 --exp-name "distill-%lang%" --early-stop 0 --step_gamma 0.94 --neg_seq_len 32
@REM python train-distill.py --accelerator "gpu" --max_epochs 64 --fasttext-path "dataset/corpus/tig/tig-ft.vec" --word2vec-path "dataset/corpus/tig/tig-w2v.vec" --corpus "dataset/corpus/tig/clean-tig-corpus.txt" --charset-path "data/am-charset.txt" --vector-load-ratio 1.0 --train-ratio 0.9 --exp-name "tig-distill-w2v-early-ft" --early-stop 1

@REM Africa
set lang=africa

@REM python train-distill.py --accelerator "gpu" --max_epochs 64 --fasttext-path "dataset/corpus/%lang%/%lang%-ft-50.vec" --word2vec-path "dataset/corpus/%lang%/%lang%-w2v-50.vec" --corpus "dataset/corpus/%lang%/clean-%lang%-corpus-50.txt" --charset-path "data/africa-charset.txt" --vector-load-ratio 0.8 --train-ratio 1.0 --exp-name "africa-distill-%lang%-50" --early-stop 0 --step_gamma 0.94 --neg_seq_len 32

python train-distill.py --accelerator "gpu" --max_epochs 64 --fasttext-path "dataset/corpus/%lang%/%lang%-ft-25.vec" --word2vec-path "dataset/corpus/%lang%/%lang%-w2v-25.vec" --corpus "dataset/corpus/%lang%/clean-%lang%-corpus-25.txt" --charset-path "data/africa-charset.txt" --vector-load-ratio 1.0 --train-ratio 1.0 --exp-name "africa-distill-%lang%-25" --early-stop 0 --step_gamma 0.94 --neg_seq_len 32
