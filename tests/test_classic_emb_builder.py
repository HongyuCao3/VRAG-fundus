import pytest
from PIL import Image
import os, json

from emb_module.emb_builder_ import ClassicEmbBuilder

@pytest.fixture(scope="module")
def classic_emb_builder():
    img_folder = "./data/Classic Images/"
    emb_folder = "./data/classic_emb_clip"
    return ClassicEmbBuilder(img_folder, emb_folder)

def test_find_similar_images(classic_emb_builder):
    input_img = "./data/level/ODIR_2450_right.jpg"
    similar_images = classic_emb_builder.find_similar_images(input_img)
    print(similar_images)
    # assert isinstance(similar_images, list), "Expected a list of tuples"
    # assert all(isinstance(t, tuple) and len(t) == 2 for t in similar_images), "Each element should be a tuple of two items"

# def test_save_image_representation(classic_emb_builder, tmpdir):
#     source_folder = "./data/Classic Images/"
#     target_folder = str(tmpdir.mkdir("representations"))
#     classic_emb_builder.save_image_representation(source_folder, target_folder)
    
#     correspondence_file = os.path.join(target_folder, 'correspondence.json')
#     assert os.path.exists(correspondence_file), "Correspondence file should exist after processing"
    
#     with open(correspondence_file, 'r') as f:
#         representation_data = json.load(f)
    
#     assert len(representation_data) > 0, "At least one image representation should have been saved"

def test_get_detailed_similarities_crop(classic_emb_builder):
    input_img = "./data/level/ODIR_2450_right.jpg"
    detailed_similarities = classic_emb_builder.get_detailed_similarities_crop(input_img)
    print(detailed_similarities)
    # assert isinstance(detailed_similarities, dict), "Expected a dictionary"
    # assert all(isinstance(v, list) for v in detailed_similarities.values()), "All values should be lists"
    # assert all(len(v) == 5 for v in detailed_similarities.values()), "Each list should contain 5 elements"