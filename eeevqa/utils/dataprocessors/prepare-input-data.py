import os
import numpy as np
import pdfkit
import pickle
import fitz
from PIL import Image, ImageOps
from collections import namedtuple

from eeevqa import read_captions, read_problem_list, read_pid_splits
from eeevqa import parse_args

# data accesor function
def get_choice_text(problem, options, verbose = False):
    choices = problem['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    if verbose:
        print(choice_txt)
    
    return choice_txt

# HTML functions
def create_question_tag():
    question_tag = "<h2>{question_content}</h2>"
    return question_tag

def create_choice_tag():
    choice_tag = "<h3>{choice_content}</h3>"
    return choice_tag

def create_img_tag():
    img_tag = '<img src="{img_source}">'
    return img_tag

def create_context_tag():
    context_tag = "<p>{context_content}</p>"
    return context_tag

def create_lecture_tag():
    lecture_tag = "<p>{lecture_content}</p>"
    return lecture_tag

def create_html_template_modular():
    html_template = \
        '''
            <html>
                <body style="background-color:white;">
                    {question_tag}
                    {choice_tag}
                    {img_tag}
                    {context_tag}
                    {lecture_tag}
                </body>
            </html>
        '''
    return html_template

def create_html_file_modular(params, df, sample_num=None):

    # create individual tags
    
    # Question
    if params["set_question_as_header"] == False:
        question_tag = create_question_tag().format(question_content = df[sample_num]["question"])
        
    else:
        question_tag = ""
        
    # Choices
    if params["set_question_as_header"] == False:
        choice_tag = create_choice_tag().format(choice_content = get_choice_text(df[sample_num], params["options"]))
        
    else:
        choice_tag = ""
        
    # Image source
    if df[sample_num]["image"] is not None:
        img_tag = create_img_tag().format(img_source = os.path.join(os.getcwd(), "data", df[sample_num]["split"], \
                              str(sample_num), df[sample_num]['image']))
    else:
        img_tag = ""

    # Context
    if df[sample_num]["hint"] is not None and params["skip_text_context"] == False:
        context_tag = create_context_tag().format(context_content = df[sample_num]["hint"])    
    else:
        context_tag = ""


    # Lecture
    if df[sample_num]["lecture"] is not None and params["skip_lecture"] == False:
        lecture_tag = create_lecture_tag().format(lecture_content = df[sample_num]['lecture']) 

    else:
        lecture_tag = ""
    
    # compose tags to html
    final_html_file = create_html_template_modular().format(question_tag = question_tag, choice_tag = choice_tag, \
                                          img_tag = img_tag, context_tag = context_tag, \
                                          lecture_tag = lecture_tag)
    
    return final_html_file

def create_html_file(html_template, df, sample_num=None):

    # Image source
    img_source = os.path.join(os.getcwd(), "data", df[sample_num]["split"], \
                              str(sample_num), df[sample_num]['image']) \
                            if df[sample_num]["image"] is not None else ""

    # Context
    context = df[sample_num]["hint"] if df[sample_num]["hint"] is not None else ""

    # Lecture
    lecture = df[sample_num]["lecture"] if df[sample_num]["lecture"] is not None else ""

    final_html_file = html_template.format(img_source = img_source, context = context, lecture = lecture)
    return final_html_file

def save_html_file(html_file, source, img_num, save_dir=""):
    with open(os.path.join(save_dir, f"{source}_{img_num}.html"), "w") as f:
        f.writelines(html_file)

# HTML to PDF
def convert_html_to_pdf(path_to_inputfile, path_to_outputfile):
    pdfkit.from_file(f'{path_to_inputfile}.html', f'{path_to_outputfile}.pdf')

# PDF to Image
def convert_pdf_to_image(path_to_inputfile, path_to_outputfile):
    
    # To get better resolution
    zoom_x = 1.0  # horizontal zoom
    zoom_y = 1.0  # vertical zoom
    mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension

    filename = f"{path_to_inputfile}.pdf"
    doc = fitz.open(filename)  # open document
    for page in doc:  # iterate through the pages
        pix = page.get_pixmap(matrix=mat)  # render page to an image
        pix.save(f"{path_to_outputfile}.jpg")  # store image as JPG

# Whitespace removal image
def remove_white_space(filename, padding = 30, visualize = False):
    
    # Open input image
    im = Image.open(filename)
    
    # Get bounding box of text and trim to it
    bbox = ImageOps.invert(im).getbbox()
    x, y = 0, 0
    x_, y_ = im.size[0], bbox[3] + padding
    new_bbox = (x, y, x_, y_)
    trimmed = im.crop(new_bbox)
    trimmed.save(filename)
    if visualize:
        trimmed.show()

# Convert Input to Image pipeline
def convert_input_to_img(problem_list, pid_splits, source="train", save_dir="", \
                         sample_subset = 10, crop_padding = 30, remove_html_file = True, remove_pdf_file = True, \
                         params=None):
    
    idx_list = pid_splits[source] 
    idx_list = [int(idx) for idx in idx_list]
    
    if sample_subset is not None:
        idx_list = idx_list[:sample_subset]
        source = "tiny_" + source
    
    # create save directory if it does not exist
    save_dir = os.path.join(save_dir, source)
        
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
        print("Save directory created")
    else:
        print("Save directory already present")
        
    
    for sample_num in idx_list:
        
        # create html template
        html_template = create_html_file_modular(params, problem_list, sample_num=sample_num)
        html_file = create_html_file(html_template, problem_list, sample_num)
        
        # save tmp html file
        save_html_file(html_file, source, sample_num, save_dir=save_dir)
        
        # tmp and final filenames
        tmp_hpi_filename = os.path.join(os.getcwd(), save_dir, f"{source}_{sample_num}")
        img_filename = os.path.join(os.getcwd(), save_dir, f"{source}_{sample_num}")
        
        # convert tmp html to tmp pdf & delete tmp html file
        convert_html_to_pdf(tmp_hpi_filename, tmp_hpi_filename)
        
        # convert tmp pdf to image & delete tmp pdf file
        convert_pdf_to_image(tmp_hpi_filename, img_filename)
        
        # crop whitespace
        remove_white_space(f"{img_filename}.jpg")
        
        # cleaning by removing tmp files
        if remove_html_file:
            os.remove(f"{tmp_hpi_filename}.html")
        
        if remove_pdf_file:
            os.remove(f"{tmp_hpi_filename}.pdf")

# Create ScienceQA dataset
def create_one_scienceqa_example(problem_list, img_filename="", sample_num=1, output_format="AE", options = None, preprocess_image=None):
    
    # header text
    header_text = problem_list[sample_num]["question"] + " " + get_choice_text(problem_list[sample_num], options)
    
    #input
    image = Image.open(f"{img_filename}.jpg")
    image_mean = 0
    image_std = 0
    if preprocess_image is not None:
        image, image_mean, image_std = preprocess_image(image)
    
    # Outputs
    if output_format == 'A':
        output = f"Answer: The answer is {problem_list[sample_num]['answer']}."
    elif output_format == 'AE':
        output = f"Answer: The answer is {problem_list[sample_num]['answer']}. BECAUSE: {problem_list[sample_num]['solution']}"
    elif output_format == 'EA':
        output = f"Answer: {problem_list[sample_num]['solution']} The answer is {problem_list[sample_num]['answer']}."

    output = output.replace("  ", " ").strip()
    if output.endswith("BECAUSE:"):
        output = output.replace("BECAUSE:", "").strip()
    
    scienceqa_example = ScienceQA(sample_num, header_text, image, image_mean, image_std, output)
    return scienceqa_example

def convert_scienceqa_to_dataset(problem_list, pid_splits, source="train", save_dir = "", output_format="AE", \
                                 options = None, preprocess_image = None, sample_subset = None):
    
    
    idx_list = pid_splits[source] 
    idx_list = [int(idx) for idx in idx_list]
    
    if sample_subset is not None:
        idx_list = idx_list[:sample_subset]
        source = "tiny_"+source
        
    save_dir = os.path.join(save_dir, source)    
    
    dataset = []
    for sample_num in idx_list:
        ifile = os.path.join(os.getcwd(), save_dir, f"{source}_{sample_num}")
        dataset.append(create_one_scienceqa_example(problem_list, img_filename=ifile, \
                                                    sample_num=sample_num, output_format=output_format, \
                                                    options = options, preprocess_image = preprocess_image))
    return dataset
        
# saving functionality
def save_dataset(dataset, save_dir="", filename=""):
    pickle_filename = os.path.join(save_dir, filename)
    with open(pickle_filename, 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    
    print("----- Parsed Arguments -----")
    args = parse_args()

    print("----- Read Dataset -----") 
    if args.task_name == "univqa":
        ScienceQA = namedtuple("ScienceQA", "sample_num header_text image image_mean image_std output")
    else:
        ScienceQA = namedtuple("ScienceQA", "sample_num header_text image text_context lecture image_mean image_std output")
        
    captions_dict = read_captions(args.data_root, args.captions_filename)
    problem_list = read_problem_list(args.json_files_path, args.problems_filename)
    pid_splits = read_pid_splits(args.json_files_path, args.pidsplits_filename)


    # set params for html
    params = {
        "set_question_as_header": args.set_question_as_header,
        "skip_text_context":args.skip_text_context,
        "skip_lecture":args.skip_lecture
    }

    # pipeline for image dataset generation
    if args.skip_image_gen == False:
        convert_input_to_img(problem_list, pid_splits, source=args.data_split,  
                        save_dir=os.path.join(os.getcwd(), args.pickle_files_path, args.data_type), sample_subset = args.sample_subset, 
                        crop_padding = args.crop_padding, params = params)
    
    # create dataset
    dataset = convert_scienceqa_to_dataset(problem_list, pid_splits, 
                      source=args.data_split, save_dir = os.path.join(os.getcwd(), 
                      args.pickle_files_path, args.data_type), 
                      output_format=args.output_format,
                      options = args.options, preprocess_image = None, 
                      sample_subset = args.sample_subset
                      ) 
    
    # save dataset
    save_dataset(
        dataset,
        save_dir = os.path.join(os.getcwd(), 
                      args.pickle_files_path, args.data_type),
        filename = f"{args.data_split}.pkl"
    )