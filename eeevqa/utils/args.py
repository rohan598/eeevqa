import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # common args
    parser.add_argument('--task_name', type=str, default='univqa')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--json_files_path', type=str, default='./data/scienceqa')
    parser.add_argument('--pickle_files_path', type=str, default='./data/new_data')
    parser.add_argument('--data_type', type=str, default='unimodal')
    parser.add_argument('--captions_filename', type=str, default='captions.json')
    parser.add_argument('--problems_filename', type=str, default='problems.json')
    parser.add_argument('--pidsplits_filename', type=str, default='pid_splits.json')
    parser.add_argument('--output_root', type=str, default='./results')
    parser.add_argument('--checkpoint_root', type=str, default='./results/checkpoints')

    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])    
    parser.add_argument('--output_format', type=str, default="AE")
    parser.add_argument('--base_model_name', type=str, default="google/pix2struct-base")
    
    # data generation args
    parser.add_argument('--skip_image_gen', type=bool, choices=[False,True], default=True)
    parser.add_argument('--data_split', type=str, default='minitrain')
    parser.add_argument('--sample_subset', default=None)
    parser.add_argument('--crop_padding', type=int, default=30)
    parser.add_argument('--set_question_as_header', type=bool, choices=[False,True], default=False)
    parser.add_argument('--skip_text_context', type=bool, choices=[False,True], default=False)
    parser.add_argument('--skip_lecture', type=bool, choices=[False,True], default=False)

    # training args
    parser.add_argument('--train_split', type=str, default='minitrain', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='minival', choices=['val', 'minival'])
    parser.add_argument('--test_split', type=str, default='minival', choices=['test', 'val', 'minival'])


    parser.add_argument('--save_every', type=int, default=1, help='Save the result with every n examples.')

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--dummy_run', type=str, default='no', choices=['yes', 'no'])

    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--max_patches',
                        type=int,
                        default=1024,
                        help='The maximum number of img patches allowed.')
    
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')


    parser.add_argument('-tbs','--train_batch_size', 
                        type=int,
                        default = 1,
                        help='training batch size')
    
    parser.add_argument('-ebs','--eval_batch_size', 
                        type=int,
                        default = 1,
                        help='evaluating batch size')
    

    parser.add_argument('-e', '--epoch', 
                        type=int,
                        default = 10,
                        help='training # epochs')
    

    parser.add_argument('-lr', '--learning_rate', 
                        type=float,
                        default = 1e-5,
                        help='learning rate')

    args = parser.parse_args()

    return args