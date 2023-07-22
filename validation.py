import sys
from diffusers.utils import load_image
from numpy import asarray

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

        image2 = run_sampler(model, asarray(image), "mmap", 1, 512)

        # image2
        cnt += 1
        metric_val += metric(image2.load(), image1.load(), 'maps-metric/colors2.json', 'maps-metric/colors.json')[0]
        print('Metric:', f'{metric_val:.2%}')
        tot_metric += metric_val

    avg_metric = tot_metric / cnt
    print('Average_metric: ', f'{avg_metric:.2%}')
    return avg_metric
