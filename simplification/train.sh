

# This line is for training solely on wikipedia data without a checkpoint
python3 train.py --config=training_config.json --model=model_config.json --save_path=wiki_model_4.pth.tar --data_source=wiki

# This line is for resuming training on wikipedia data with a checkpoint
# It can happen because running out gpu quota, restart with a checkpoint
#python3 train.py --checkpoint=wiki_model.pth.tar --config=training_config.json \
#--model=model_config.json --save_path=wiki_model.pth.tar --data_source=wiki

# This line is for training solely on md data without a checkpoint
#python3 train.py --config=training_config.json --model=model_config.json --save_path=/kaggle/working/md_mode.pth.tar --data_source=md

# This line is for transfer learning on md data
#python3 train.py --checkpoint=wiki_model.pth.tar --config=training_config.json --model=model_config.json \
#--save_path=transfer.pth.tar --data_source=md