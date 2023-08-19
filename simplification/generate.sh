

# This line is for generating wiki simplification on the wiki-trained model
python3 generate.py --model=wiki_model.pth.tar --path=../simplification_data/test.src --beam=5 \
 --to_path=../simplification_data/test1.gen

# This line is for generating md simplification on the md-trained model
#python3 generate.py --model=md_model.pth.tar --path=../md_simplification_data/test.src --beam=5 \
#--to_path=../md_simplification_data/test1.gen

