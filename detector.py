import argparse
import pickle
from collections import Counter
from pathlib import Path
#kys
#kys again
import face_recognition
from PIL import Image, ImageDraw,  ImageFont

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "green"
TEXT_COLOR = "white"
FONT_PATH_1=rf"{Path.cwd()}\ARIAL.TTF"
FONT_PATH_2=rf"{Path.cwd()}\COMiCSANS.TTF"

# Create directories if they don't already exist
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument(
    "--validate", action="store_true", help="Validate trained model"
)
parser.add_argument(
    "--test", action="store_true", help="Test the model with an unknown image"
)
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)
parser.add_argument(
    "-f", action="store", help="Path to an image with an unknown face"
)
args = parser.parse_args()


def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    """
    Loads images in the training directory and builds a dictionary of their
    names and encodings.
    """
    names = []
    encodings = []
    counts = {}

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)
        
        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)
            if name in counts:
                counts[name] += 1
            else:
                counts[name] = 1

    name_encodings = {"names": names, "encodings": encodings,"counts":counts}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    """
    Given an unknown image, get the locations and encodings of any faces and
    compares them against the known encodings to find potential matches.
    """
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)
    image_width, image_height = pillow_image.size
    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name, common , total = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, f"{name} {common}/{total}",(image_width, image_height))
    del draw
    pillow_image.show()


def _recognize_face(unknown_encoding, loaded_encodings):
    """
    Given an unknown encoding and all known encodings, find the known
    encoding with the most matches.
    """
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        most_common_vote, most_common_count = votes.most_common(1)[0]
        total_votes = loaded_encodings["counts"][most_common_vote]
        return most_common_vote, most_common_count , total_votes
    return None, 0,0


def _display_face(draw, bounding_box, name,image_size):
    """
    Draws bounding boxes around faces, a caption area, and text captions.
    """
    image_width,image_height=image_size
    font_size = max(10, int(image_width * 0.035))  
    box_width = max(1, int(image_width * 0.01))
    
    font1 = ImageFont.truetype(FONT_PATH_2, font_size) 
    
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR, width= box_width)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        
        (left, bottom), name,font=ImageFont.truetype(FONT_PATH_2, font_size*1.3) 

    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill=BOUNDING_BOX_COLOR,
        outline=BOUNDING_BOX_COLOR,
    )
    draw.text(
        (text_left, text_top),
        name,
        fill=TEXT_COLOR,
        font= font1
    )


def validate(model: str = "hog"):
    """
    Runs recognize_faces on a set of images with known faces to validate
    known encodings.
    """
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )


if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)
