"# bpe_algorithm_can_finetune_tokenizer" 

this is an implyment for https://github.com/huggingface/transformers/issues/15153


I just add tens of lines of code into the py_bpe algorithm.
function finetune_tokenizer is main function added.


Details can be see in example.py , actuctally it is very simple.
the official python library tokenizer is written is rust. I am learning hoping to give a rust version of this code.



ps:
the_factor_of_new_added_token_divided_unk_number is the only param you should set.
hoping can find a auto algorithm to set it.













