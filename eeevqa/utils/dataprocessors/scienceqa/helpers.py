import os
import fitz
from PIL import Image, ImageOps, ImageDraw, ImageFont
import textwrap
import pdfkit
import torchvision.transforms as transforms
from collections import namedtuple
import math
import pickle

######### Common Methods #########

def get_specific_split(idx_list, problem_list, key_attr=""):
    final_idx_list = []
    for i in idx_list:
        if problem_list[i][key_attr] is None:
            final_idx_list.append(i)
    return final_idx_list

def input_to_image_initialization(problem_list, pid_splits, params=None):
    
    data_split = params["data_split"]
    idx_list = pid_splits[data_split] 
    idx_list = [int(idx) for idx in idx_list]
    
    sample_subset = params["sample_subset"]
    if sample_subset is not None:
        try:
            sample_subset = int(sample_subset)
            idx_list = idx_list[:params["sample_subset"]]
            data_split = "tiny_" + data_split
        except:
            idx_list = get_specific_split(idx_list, problem_list, key_attr=sample_subset)
            data_split = sample_subset + "_" +  data_split

    
    # create save directory if it does not exist

    save_dir = os.path.join(params["save_dir"], data_split)
    print(save_dir)        
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
        print("Save directory created")
    else:
        print("Save directory already present")
    
    stats_dict = {
        "original_size_w":[],
        "original_size_h":[],
        "final_size_w":[],
        "final_size_h":[]
    }

    return data_split, idx_list, save_dir, stats_dict

######### HTML to Image Processing Methods #########
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


def create_html_template_modular(tags=[]):
    html_template = '<html> <body style="background-color:white;">'
    for tag in tags:
        html_template += "{" + tag + "}"           
    html_template+="</body></html>"

    return html_template

def create_html_file_modular(params, df, sample_num=None, layout_type=1):

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
    if layout_type==1: # QM I CL
        tags = ["question_tag", "choice_tag", "img_tag", "context_tag", "lecture_tag"]
        
    elif layout_type==2: # QM CL I
        tags = ["question_tag", "choice_tag", "context_tag", "lecture_tag", "img_tag"]
    
    elif layout_type==3: # QM I
        tags = ["question_tag", "choice_tag", "img_tag"]
    
    
    final_html_file = create_html_template_modular(tags).format(question_tag = question_tag, choice_tag = choice_tag, \
                                          img_tag = img_tag, context_tag = context_tag, \
                                          lecture_tag = lecture_tag)
    
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
    with fitz.open(filename) as doc:
        for page in doc:  # iterate through the pages
            pix = page.get_pixmap(matrix=mat)  # render page to an image
            pix.save(f"{path_to_outputfile}.jpg")  # store image as JPG

# Whitespace removal image
def remove_white_space(filename, crop_padding = 30, visualize = False, stats_dict = None):
    
    # Open input image
    im = Image.open(filename)
    
    # Get bounding box of text and trim to it
    bbox = ImageOps.invert(im).getbbox()
    x, y = 0, 0
    x_, y_ = im.size[0], bbox[3] + crop_padding
    new_bbox = (x, y, x_, y_)
    trimmed = im.crop(new_bbox)
    trimmed.save(filename)
    if visualize:
        trimmed.show()
    
    stats_dict["final_size_w"].append(trimmed.size[0])
    stats_dict["final_size_h"].append(trimmed.size[1])

    return stats_dict
    

######### Render Text on Image Processing Methods #########

def render_text_on_image(image, header, text_context, params):
    """Renders a header and text context on a PIL image and returns a new PIL image."""
    params["is_text_context"] = False
    header_image = render_text(header, params)
    layout_type = params["layout_type"]
    if text_context!="":
        params["is_text_context"] = True
        text_context_image = render_text(text_context, params)
        
    else:
        layout_type = 3
        text_context_image = Image.new("RGB",(1, 1),"white")
    
    new_width = max(max(header_image.width, text_context_image.width), image.width)
        
    new_height = image.height if image.height == 1 else int(image.height *  (new_width / image.width))
    
    new_header_height = int(
      header_image.height * (new_width / header_image.width))
    
    new_text_context_height = text_context_image.height if text_context_image.height == 1 else int(
      text_context_image.height * (new_width / text_context_image.width))
    
    new_image = Image.new(
      "RGB",
      (new_width, new_height + new_header_height + new_text_context_height),
      "white")

    if layout_type==1: # QM I CL

        new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))
        new_image.paste(image.resize((new_width, new_height)), (0, new_header_height))
        new_image.paste(text_context_image.resize((new_width, new_text_context_height)), (0, new_header_height+new_height))
        
    elif layout_type==2: # QM CL I
        new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))
        new_image.paste(text_context_image.resize((new_width, new_text_context_height)), (0, new_header_height))
        new_image.paste(image.resize((new_width, new_height)), (0, new_header_height+new_text_context_height))
        
    elif layout_type==3: # QM I
        new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))
        new_image.paste(image.resize((new_width, new_height)), (0, new_header_height))
        
    return new_image


def render_text(text,
                params):
    
    """Render text."""
    # Add new lines so that each line is no more than 80 characters.
    wrapper = textwrap.TextWrapper(width=80)
    lines = wrapper.wrap(text=text)
    wrapped_text = "\n".join(lines)
    if params["is_text_context"] == False:
        font = ImageFont.truetype(os.path.join(params["path_to_font"], "arial.ttf"), encoding="UTF-8", size=params["header_size"])
    else:
        font = ImageFont.truetype(os.path.join(params["path_to_font"], "arial.ttf"), encoding="UTF-8", size=params["text_context_size"])
    
    # Use a temporary canvas to determine the width and height in pixels when
    # rendering the text.
    temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1), params["background_color"]))
    _, _, text_width, text_height = temp_draw.textbbox((0, 0), wrapped_text, font)

    # Create the actual image with a bit of padding around the text.
    image_width = text_width + params["left_padding"] + params["right_padding"]
    image_height = text_height + params["top_padding"] + params["bottom_padding"]
    image = Image.new("RGB", (image_width, image_height), params["background_color"])
    draw = ImageDraw.Draw(image)
    
    draw.text(
      xy=(params["left_padding"], params["top_padding"]),
      text=wrapped_text,
      fill=params["text_color"],
      font=font)
    
    return image  

def image_creator(image_dir, problem_list, sample_num=1, stats_dict=None, params=None):
    
    # Open input image
    if os.path.exists(os.path.join(image_dir, str(sample_num), "image.png")):
        im = Image.open(os.path.join(image_dir, str(sample_num), "image.png"))
        if params["visualize"]:
            im.show()
        
    else:
        im = Image.new("RGB",(1, 1),"white")
    
    if params["analyze"]:
        stats_dict["original_size_w"].append(im.size[0])
        stats_dict["original_size_h"].append(im.size[1])
    
    # combine question and choice, strip all leading and trailing spaces, 
    # remove all newlines
    question_text = problem_list[sample_num]["question"]
    choice_text = get_choice_text(problem_list[sample_num], params["options"])
    question_choice_text = question_text + " " + choice_text
    
    # combine lecture and text context, strip all leading and trailing spaces, 
    # remove all newlines

    # Context
    if problem_list[sample_num]["hint"] is not None and params["skip_text_context"] == False:
        hint_text = ' '.join(problem_list[sample_num]["hint"].split("/n")) 
    else:
        hint_text = ""


    # Lecture
    if problem_list[sample_num]["lecture"] is not None and params["skip_lecture"] == False:
        lecture_text = ' '.join(problem_list[sample_num]["lecture"].split("/n"))

    else:
        lecture_text = ""
        
    hint_lecture_text = (hint_text + " " + lecture_text).strip()
    
    final_im = render_text_on_image(im, question_choice_text, hint_lecture_text, params)
    
    if params["analyze"]:
        final_w = final_im.size[0]
        final_h = final_im.size[1]
        if final_w * final_h > 4096*256:

            max_ph = int(math.ceil((4096*256)/final_w))
            test_resize = final_im.copy()
            test_resize = test_resize.resize((final_w, max_ph))
            
            test_clip = final_im.copy()
            test_clip = test_clip.crop((0, 0, final_w, max_ph))

        
        stats_dict["final_size_w"].append(final_im.size[0])
        stats_dict["final_size_h"].append(final_im.size[1])
    
    if params["visualize"]:
        final_im.show()
        
    if params["save"]:
        final_im.save(params["save_path"])
        
    return stats_dict

######### ScienceQA Dataset methods #########
# Create ScienceQA dataset
def create_one_scienceqa_example(text_data, img_filename="", sample_num=1, max_patches = 4096, processor=None, TrainQA = None):
    
    #input
    image = Image.open(f"{img_filename}.jpg")
    encoding = processor(images=image, return_tensors="pt", add_special_tokens=True, max_patches=max_patches)
    encoding = {k:v.squeeze() for k,v in encoding.items()} 
    flattened_patches = encoding["flattened_patches"]
    attention_mask = encoding["attention_mask"]

    raw_output = text_data["raw_output"][sample_num]
    output = text_data["targets"].input_ids[sample_num]
    
    scienceqa_example = TrainQA(sample_num, image, flattened_patches, attention_mask, raw_output, output)

    return scienceqa_example

# saving functionality
def save_dataset(dataset, save_dir="", filename=""):
    pickle_filename = os.path.join(save_dir, filename)
    with open(pickle_filename, 'wb') as f:
        pickle.dump(dataset, f)