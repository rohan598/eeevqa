import argparse

def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False

def parse_args():
    parser = argparse.ArgumentParser()

    # common args
    parser.add_argument('--task_name', type=str, default='univqa')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--data_version', type=str, default='v1')
    parser.add_argument('--json_files_dir', type=str, default='scienceqa')
    parser.add_argument('--pickle_files_dir', type=str, default='new_data')
    parser.add_argument("--assets_dir", type=str, default="eeevqa/assets")

    parser.add_argument('--data_type', type=str, default='unimodal')
    parser.add_argument('--layout_type', type=int, default=1)
    parser.add_argument('--captions_filename', type=str, default='captions.json')
    parser.add_argument('--problems_filename', type=str, default='problems.json')
    parser.add_argument('--pidsplits_filename', type=str, default='pid_splits.json')
    parser.add_argument('--output_root', type=str, default='results')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])    
    parser.add_argument('--output_format', type=str, default="AE")
    parser.add_argument('--base_model_name', type=str, default="google/pix2struct-base")
    parser.add_argument('--gpu_cnt', type=int, default=1)


    # data generation args
    parser.add_argument('--skip_image_gen', type=str, default="False")
    parser.add_argument('--skip_dataset_gen', type=str, default="False")
    parser.add_argument('--data_split', type=str, default='minitrain')
    parser.add_argument('--data_source', type=str, default='train')
    parser.add_argument('--sample_subset', type=str, default="")
    parser.add_argument('--skip_text_context', type=str, default="False")
    parser.add_argument('--skip_lecture', type=str, default="False")
    parser.add_argument('--visualize_gen', type=str, default="False")

    ## data gen HTML to Image Technique
    parser.add_argument('--crop_padding_dgv1', type=int, default=30)
    parser.add_argument('--set_question_as_header_dgv1', type=str, default="False")
    parser.add_argument('--remove_html_file_dgv1', type=str, default="True")
    parser.add_argument('--remove_pdf_file_dgv1', type=str, default="True")


    ## data gen render text on image
    parser.add_argument('--text_color_dgv2', type=str, default="black")
    parser.add_argument('--background_color_dgv2', type=str, default="white")
    parser.add_argument('--header_size_dgv2', type=int, default=36)
    parser.add_argument('--text_context_size_dgv2', type=int, default=18)
    parser.add_argument('--top_padding_dgv2', type=int, default=5)
    parser.add_argument('--right_padding_dgv2', type=int, default=5)
    parser.add_argument('--bottom_padding_dgv2', type=int, default=5)
    parser.add_argument('--left_padding_dgv2', type=int, default=5)
    parser.add_argument('--analyze_dgv2', type=str, default="False")


    # training args
    parser.add_argument('--train_split', type=str, default='minitrain', choices=['train', 'trainval', 'minitrain','image_train', 'image_minitrain'])
    parser.add_argument('--val_split', type=str, default='minival', choices=['val', 'test', 'minival', 'image_val','image_minival'])
    parser.add_argument('--test_split', type=str, default='image_minitest', choices=['test', 'minitest','image_test','image_minitest'])


    parser.add_argument('--save_every_n_epoch', type=int, default=1, help='model checkpoint with every n epochs.')

    # parser.add_argument('--save_every_n_steps', type=int, default=1, help='model checkpoint with every n epochs.')

    # parser.add_argument('--log_every_n_epoch', type=int, default=1, help='model checkpoint with every n epochs.')

    parser.add_argument('--log_every_n_steps', type=int, default=1, help='model checkpoint with every n epochs.')

    parser.add_argument('--es_monitor', type=str, default="val_acc", help='Early stopping monitor metric')

    parser.add_argument('--es_mode', type=str, default="max", help='Early stopping monitor metric comparison')

    parser.add_argument('--es_min_delta', type=float, default=0.0, help='Early stopping monitor metric minimum required change')

    parser.add_argument('--es_patience', type=int, default=3, help='Early stopping monitor metric max epochs without improvement')

    parser.add_argument('--dummy_run', type=str, default='no', choices=['yes', 'no'])

    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--max_patches',
                        type=int,
                        default=4096,
                        help='The maximum number of img patches allowed.')
    
    parser.add_argument('--max_new_tokens',
                        type=int,
                        default=128,
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
    
    # parser.add_argument('-wd', '--weight_decay', 
    #                     type=float,
    #                     default = 1e-5,
    #                     help='weight decay')

    parser.add_argument('--skip_scheduler', type=str, default="True")
    
    parser.add_argument('--warmup_steps', 
                            type=int,
                            default = 1000,
                            help='warmup steps in linear warmup with cosine anneal schedule')
    
    parser.add_argument('--total_steps', 
                            type=int,
                            default = 10000,
                            help='total number of steps in linear warmup with cosine anneal schedule')
    
    parser.add_argument('--cycles', 
                            type=float,
                            default = 0.5,
                            help='cosine decay cycles')

    parser.add_argument('--num_workers', 
                        type=int,
                        default = 8,
                        help='number of cpu workers for dataloading')
    
    # evaluation args
    parser.add_argument('--eval_split', type=str, default='minival', choices=['minitest', 'test', 'val', 'minival','tiny_test','tiny_val',"train","minitrain","tiny_train"])

    parser.add_argument('--eval_checkpoint_name', type=str, default='')

    parser.add_argument('--results_dir', type=str, default='model_results')

    parser.add_argument('--skip_model_output_gen', type=str, default="False")
    
    parser.add_argument('--skip_model_score_pred', type=str, default="False")
    
    args = parser.parse_args()

    return args 