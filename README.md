# cs374-final
I have `abilities.csv`, `cards.csv`, `draft_data_public.WOE.PremierDraft.csv`, `game_data_public.WOE.PremierDraft.csv`, and `replay_data_public.WOE.PremierDraft.csv` files in the directory, but they are too big for github.

# Notes
Cleaning the full dataset output: 153878 usable drafts (580392 lines discarded)

# Old Notes
1,030,700 parameter model (training) 10 epochs
R^2: 0.8768292665481567

1,030,700 parameter model (testing)
R^2: -0.3145120143890381

1,030,700 parameter model with dropout (training) 10 epochs
R^2: -0.6828773021697998

1,030,700 parameter model with dropout (testing)
R^2: -0.8698959350585938

330 parameter linear regression (training) (trained with SGD) 30 epochs
R^2: -1.217928171157837

330 parameter linear regression (testing)
R^2: -1.1490283012390137

53,100 parameter model with dropout (training) 10 epochs
R^2: -1.5023865699768066

53,100 parameter model with dropout (testing)
R^2: -1.5453803539276123