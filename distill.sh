

lang=africa

python train-distill.py --accelerator "gpu" --max_epochs 64 --fasttext-path "dataset/corpus/$lang/$lang-ft-50.vec" --word2vec-path "dataset/corpus/$lang/$lang-w2v-50.vec" --corpus "dataset/corpus/$lang/clean-$lang-corpus-50.txt" --charset-path "data/africa-charset.txt" --vector-load-ratio 1.0 --train-ratio 1.0 --exp-name "africa-distill-$lang-50" --early-stop 0 --step_gamma 0.94 --neg_seq_len 32
