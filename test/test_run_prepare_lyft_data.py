from dataset.run_prepare_lyft_data import SceneProcessor

from helpers.config_tool import get_paths

_, artifact_path, _ = get_paths()

sp = SceneProcessor("train", from_rgb_detection=False)

numbers = sp.find_processed_scenes(artifact_path)

print(numbers)
