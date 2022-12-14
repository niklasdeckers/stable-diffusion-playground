import argparse
import base64
import os
import time
from io import BytesIO
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from utils import parse_arg_boolean

app = Flask(__name__)
CORS(app)
print("--> Starting server. This might take a few minutes.")

model = None

parser = argparse.ArgumentParser(description="An app to turn your textual prompts into visionary delights")
parser.add_argument("--port", type=int, default=8000, help="backend port")
parser.add_argument("--model_path", type=str, help="Path of the model.ckpt")
parser.add_argument("--save_to_disk", type=parse_arg_boolean, default=False,
                    help="Should save generated images to disk")
parser.add_argument("--img_format", type=str.lower, default="JPEG", help="Generated images format",
                    choices=['jpeg', 'png'])
parser.add_argument("--output_dir", type=str, default="./", help="Customer directory for generated images")
args = parser.parse_args()


@app.route("/backend", methods=["POST"])
@cross_origin()
def generate_images_api():
    json_data = request.get_json(force=True)
    text_prompt = json_data["text"]
    num_images = json_data["num_images"]
    seed = json_data.get("seed", -1)
    generated_imgs = model.generate_images(text_prompt, num_images, seed)

    returned_generated_images = []
    if args.save_to_disk:
        dir_name = os.path.join(args.output_dir, f"{time.strftime('%Y-%m-%d_%H:%M:%S')}_{text_prompt}")
        Path(dir_name).mkdir(parents=True, exist_ok=True)

    for idx, img in enumerate(generated_imgs):
        if args.save_to_disk:
            img.save(os.path.join(dir_name, f'{idx}.{args.img_format}'), format=args.img_format)

        buffered = BytesIO()
        img.save(buffered, format=args.img_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        returned_generated_images.append(img_str)

    print(f"Created {num_images} images from text prompt [{text_prompt}]")

    response = {'generatedImgs': returned_generated_images,
                'generatedImgsFormat': args.img_format}
    return jsonify(response)


@app.route("/", methods=["GET"])
@cross_origin()
def health_check():
    return jsonify(success=True)


from stable_diffusion_model import Model

with app.app_context():
    model = Model(args.model_path)
    model.generate_images("warm-up", 1)
    print("--> Server is up and running!")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=False)
