import os
import random
from PIL import Image
import json
import shutil
from tqdm import tqdm
import sys

TARGET_PATH = 'cn'
BASE_PATH = 'dataset'
SEED = 42
IMG_CNT = 25
TARGET_SIZE = 512
FULL_SIZE = 600
PROMPT = 'mmap'

def main():
	try:
		images = os.listdir(BASE_PATH)
	except FileNotFoundError:
		print('Dataset not found')
		return

	os.makedirs(TARGET_PATH, exist_ok=True)
	shutil.rmtree(TARGET_PATH)
	os.mkdir(TARGET_PATH)
	os.mkdir(os.path.join(TARGET_PATH, 'source'))
	os.mkdir(os.path.join(TARGET_PATH, 'target'))

	F = open(os.path.join(TARGET_PATH, 'prompt.json'), 'w')

	for f in tqdm(images):
		try:
			img = Image.open(os.path.join(BASE_PATH, f))
			img_source = img.crop((0, 0, TARGET_SIZE, TARGET_SIZE))
			img_target = img.crop((FULL_SIZE, 0, FULL_SIZE + TARGET_SIZE, TARGET_SIZE))
			img_source.save(os.path.join(TARGET_PATH, 'source', f))
			img_target.save(os.path.join(TARGET_PATH, 'target', f))
			F.write(json.dumps({'source': os.path.join('source', f), 'target': os.path.join('target', f), 'prompt': PROMPT}) + '\n')
		except:
			print(f"Can't open {f}")
		sys.exit(0)

if __name__ == '__main__':
	main()

