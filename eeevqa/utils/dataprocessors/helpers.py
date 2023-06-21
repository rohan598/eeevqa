import os
import fitz
from PIL import Image, ImageOps, ImageDraw, ImageFont
import textwrap
import pdfkit
import torchvision.transforms as transforms
from collections import namedtuple
import math

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

######### Render Text on Image Processing Methods #########

def render_text_on_image(image, header, text_context, params):
    """Renders a header and text context on a PIL image and returns a new PIL image."""
    params["is_text_context"] = False
    header_image = render_text(header, params)
    image_type = params["image_type"]
    print("tc: ", r"{}".format(text_context))
    if text_context!="":
        params["is_text_context"] = True
        print(type(text_context))
        text_context_image = render_text(text_context, params)
        print("here")
        
    else:
        image_type = 3
        text_context_image = Image.new("RGB",(1, 1),"white")
        print(text_context_image.size, text_context_image.height)

#     print(f"image, header, text context = {image.size}, {header_image.size}, {text_context_image.size}")
    
    new_width = max(max(header_image.width, text_context_image.width), image.width)
        
    new_height = image.height if image.height == 1 else int(image.height *  (new_width / image.width))
    
    new_header_height = int(
      header_image.height * (new_width / header_image.width))
    
    new_text_context_height = text_context_image.height if text_context_image.height == 1 else int(
      text_context_image.height * (new_width / text_context_image.width))
    
    print(new_text_context_height)
    new_image = Image.new(
      "RGB",
      (new_width, new_height + new_header_height + new_text_context_height),
      "white")

    if image_type==1: # QM I CL

        new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))
        new_image.paste(image.resize((new_width, new_height)), (0, new_header_height))
        new_image.paste(text_context_image.resize((new_width, new_text_context_height)), (0, new_header_height+new_height))
        
    elif image_type==2: # QM CL I
        new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))
        new_image.paste(text_context_image.resize((new_width, new_text_context_height)), (0, new_header_height))
        new_image.paste(image.resize((new_width, new_height)), (0, new_header_height+new_text_context_height))
        
    elif image_type==3: # QM I
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
    choice_text = get_choice_text(problem_list[sample_num], options)
    question_choice_text = question_text + " " + choice_text
    print(question_choice_text)
    # combine lecture and text context, strip all leading and trailing spaces, 
    # remove all newlines
    hint_text = ' '.join(problem_list[sample_num]["hint"].split("/n"))
    lecture_text = ' '.join(problem_list[sample_num]["lecture"].split("/n"))
    hint_lecture_text = (hint_text + " " + lecture_text).strip()
    print(hint_lecture_text)
    
    final_im = render_text_on_image(im, question_choice_text, hint_lecture_text, params)
    
    if params["analyze"]:
        final_w = final_im.size[0]
        final_h = final_im.size[1]
        if final_w * final_h > 4096*256:
            print(final_w * final_h)
            final_im.show()
            max_ph = int(math.ceil((4096*256)/final_w))
            test_resize = final_im.copy()
            test_resize = test_resize.resize((final_w, max_ph))
            test_resize.show()
            
            test_clip = final_im.copy()
            test_clip = test_clip.crop((0, 0, final_w, max_ph))
            test_clip.show()
            stats_dict["ret"] = True
            return stats_dict
        stats_dict["final_size_w"].append(final_im.size[0])
        stats_dict["final_size_h"].append(final_im.size[1])
    
    if params["visualize"]:
        final_im.show()
        
    if params["save"]:
        final_im.save(params["save_path"])
        
    return stats_dict

######### ScienceQA Dataset methods #########
# Create ScienceQA dataset
def create_one_scienceqa_example(problem_list, img_filename="", sample_num=1, output_format="AE", options = None, preprocess_image=None, task_name=None):
    
    # header text
    header_text = problem_list[sample_num]["question"] + " " + get_choice_text(problem_list[sample_num], options)

    text_context =  problem_list[sample_num]["hint"] 
    lecture =  problem_list[sample_num]["lecture"] 
    
    #input
    pil_to_tensor_transform = transforms.Compose([
    transforms.PILToTensor()
    ])
    with Image.open(f"{img_filename}.jpg") as img:
        image = pil_to_tensor_transform(img)
        image_mean = 0
        image_std = 0
        if preprocess_image is not None:
            image, image_mean, image_std = preprocess_image(image)

    # Outputs
    if output_format == 'A':
        output = f"Answer: The answer is {options[problem_list[sample_num]['answer']]}."
    elif output_format == 'AE':
        output = f"Answer: The answer is {options[problem_list[sample_num]['answer']]}. BECAUSE: {problem_list[sample_num]['solution']}"
    elif output_format == 'EA':
        output = f"Answer: {problem_list[sample_num]['solution']} The answer is {options[problem_list[sample_num]['answer']]}."

    output = output.replace("  ", " ").strip()
    if output.endswith("BECAUSE:"):
        output = output.replace("BECAUSE:", "").strip()
    
    if task_name == "univqa":
        scienceqa_example = ScienceQA(sample_num, header_text, image, image_mean, image_std, output)
    else:
        scienceqa_example = ScienceQA(sample_num, header_text, image, text_context, lecture, image_mean, image_std, output)

    return scienceqa_example

# dummy main to prevent error cause by ScienceQA definition used int he functions
if __name__ == '__main__':
    
    task_name = "univqa"
    if task_name == "univqa":
        ScienceQA = namedtuple("ScienceQA", "sample_num header_text image image_mean image_std output")
    else:
        ScienceQA = namedtuple("ScienceQA", "sample_num header_text image text_context lecture image_mean image_std output")