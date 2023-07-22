import sys
from diffusers.utils import load_image
from numpy import asarray
from PIL import Image
import os
from cldm.model import create_model, load_state_dict

sys.path.insert(1, 'maps-metric')

from fast_metric import metric
from inference import run_sampler

def validate_model(model):
    print('Running metric')
    tot_metric = 0
    cnt = 0
    BASE_PATH = 'maps-metric/tokyo5'

    for filename in os.listdir(BASE_PATH):
        print(f'Iteration {cnt}, file {filename}')
        image0 = load_image(f'{BASE_PATH}/{filename}')
        image = image0.crop([0, 0, 512, 512])
        image1 = image0.crop([600, 0, 600 + 512, 512])
        num_samples = 1

        image2_arr = run_sampler(model, asarray(image), "mmap", 1, 512)[0]
        image2 = Image.fromarray(image2_arr)

        cnt += 1
        metric_val = metric(image2, image1, 'maps-metric/colors2.json', 'maps-metric/colors.json')[0]
        print('Metric:', f'{metric_val:.2%}')
        tot_metric += metric_val

    avg_metric = tot_metric / cnt
    print('Average_metric: ', f'{avg_metric:.2%}')
    return avg_metric

if __name__ == '__main__':
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('100.ckpt', location='cpu'))
    validate_model(model)
