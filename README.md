# README

To run the code, run:

> python main.py 

You will need to download the datasets yourself from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html.
Default dataset location is "./dataset". You can change the --dpath default values as you want.

To load pretrained model, run:
> python main.py --restore_ckpt ./checkpoints/model_name.pth

To test yout pretrained model and get .csv result file, run:
> python main.py --restore_ckpt ./checkpoints/model_name.pth --test_mode 1 

Do not change nway, kshot, query numbers!

Do not change nway, kshot, query numbers! 

