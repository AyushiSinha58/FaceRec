import argparse
import pickle
from collections import Counter
from pathlib import Path

import face_recognition
from PIL import Image, ImageDraw,  ImageFont

enc_path = Path("output/encodings.pkl")
BOX_COLOR = "green"
TEXT_COLOR = "white"
FONT_PATH_1=rf"{Path.cwd()}\ARIAL.TTF"
FONT_PATH_2=rf"{Path.cwd()}\COMiCSANS.TTF"


Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument(
    "--validate", action="store_true", help="check if your model is working or not"
)
parser.add_argument("--test", action="store_true", help="you can use this for checking with any particular image stored on your device "
)
parser.add_argument(
    "-modes",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="two modes are available , cnn for devices which have a GPU and cnn for devices without a cpu",
)
parser.add_argument(
    "-file", action="store", help="add a path to an image"
)
args = parser.parse_args()


def encode_faces(
    model: str = "hog", encodings_location: Path = enc_path
) -> None:

    names = []
    encodings = []
    counts = {}

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)
        
        f_locations = face_recognition.face_locations(image, model=model)
        f_encodings = face_recognition.face_encodings(image, f_locations)

        for encoding in f_encodings:
            names.append(name)
            encodings.append(encoding)
            if name in counts:
                counts[name] += 1
            else:
                counts[name] = 1

    name_encoding = {"names": names, "encodings": encodings,"counts":counts}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encoding, f)


def rec_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = enc_path,
) -> None:
 
    with encodings_location.open(mode="rb") as f:
        loaded_enc = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_f_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_f_encodings = face_recognition.face_encodings(
        input_image, input_f_locations
    )

    pil_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pil_image)
    image_width, image_height = pil_image.size
    for bounding_box, unknown_encoding in zip(
        input_f_locations, input_f_encodings
    ):
        name, common , total = _rec_face(unknown_encoding, loaded_enc)
        percentage=int(common/total*100)

        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, f"{name} {percentage}%",(image_width, image_height))
    del draw
    pil_image.show()


def _rec_face(unknown_encoding, loaded_enc):
    boolean_matches = face_recognition.compare_faces(
        loaded_enc["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_enc["names"])
        if match
    )
    if votes:
        most_common_vote, most_common_count = votes.most_common(1)[0]
        total_votes = loaded_enc["counts"][most_common_vote]
        return most_common_vote, most_common_count , total_votes
    return None, 0,0


def _display_face(draw, bounding_box, name,image_size):

    image_width,image_height=image_size
    font_size = max(10, int(image_width * 0.035)) 
    box_width = max(1, int(image_width * 0.01))
    
    font1 = ImageFont.truetype(FONT_PATH_2, font_size) 
    
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOX_COLOR, width= box_width)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        
        (left, bottom), name,font=ImageFont.truetype(FONT_PATH_2, font_size*1.3) 

    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill=BOX_COLOR,
        outline=BOX_COLOR,
    )
    draw.text(
        (text_left, text_top),
        name,
        fill=TEXT_COLOR,
        font= font1
    )


def validate(model: str = "hog"):

    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            rec_faces(
                image_location=str(filepath.absolute()), model=model
            )


if __name__ == "__main__":
    if args.train:
        encode_faces(model=args.modes)
    if args.validate:
        validate(model=args.modes)
    if args.test :
        rec_faces(image_location=args.file, model=args.modes)
