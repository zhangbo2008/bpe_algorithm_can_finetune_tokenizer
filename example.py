import tqdm
from py_bpe import BpeTokenizer
from pathlib import Path
savepath = Path("penguin_of_doom.vocab")
corpus = """
    hi every1 im new!!!!!!! *holds up spork* my name is katy but u can call me t3h PeNgU1N oF d00m!!!!!!!! lol…as u can see im very random!!!! thats why i came here, 2 meet random ppl like me ^_^… im 13 years old (im mature 4 my age tho!!) i like 2 watch invader zim w/ my girlfreind (im bi if u dont like it deal w/it) its our favorite tv show!!! bcuz its SOOOO random!!!! shes random 2 of course but i want 2 meet more random ppl =) like they say the more the merrier!!!! lol…neways i hope 2 make alot of freinds here so give me lots of commentses!!!!
    DOOOOOMMMM!!!!!!!!!!!!!!!! <--- me bein random again ^_^ hehe…toodles!!!!!
    love and waffles,
    t3h PeNgU1N oF d00m
"""

learn_bpe_args = dict(
    vocab_size=1000,
    pairable_chars="a-zA-Z0-9",
)

bpet = BpeTokenizer.from_corpus(corpus, savepath, learn_bpe_args=learn_bpe_args)
unk_char = "%"
tokens = bpet.tokenize("t3h PeNgU1N oF d00m"+unk_char)
print(tokens)

finetune_corpus='''hi every1 im new sssdlaj ssdsajlfk ssdsafjkl的斯拉克福建烤老鼠大解放路卡啥的'''
token_before_finetune=bpet.encode(finetune_corpus)
print(token_before_finetune)#[22, 22, 22, 25, 23, 18, 0, 12, 22, 22, 123, 18, 0, 23, 28, 33, 12, 22, 22, 123, 220, 0, 33, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print('we see there are too many zero means unk')
#==========================adding code for extension: finetune a new tokenizer
the_factor_of_new_added_token_divided_unk_number=1.5 # we set this factor because, we expand our tokenizer, so the new corpus must be many unk with old tokenizer, our new_tokenizer length of need a new_added_token, so we set a factor, if the factor is higher, we have more new tokenizer. the factor must be bigger than 1.0
new_tokenizer=bpet.finetune_tokenizer(finetune_corpus,the_factor_of_new_added_token_divided_unk_number)

token_after_finetune=new_tokenizer.encode(finetune_corpus)
print(token_after_finetune)#[239, 240, 244, 223, 12, 239, 123, 241, 246, 33, 12, 239, 123, 220, 223, 254, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 224]














