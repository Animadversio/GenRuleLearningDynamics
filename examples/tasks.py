"""Small CPU example: discrete rules, verbatim language prompts and optional images."""
import argparse
import numpy as np
from genrule.rules.parity import sample_group_parity_vec, parity_func
from genrule.datasets.language.light_switch import SwitchTask, render_prompt, sample_valid, format_answer, verify


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--images',action='store_true');args=ap.parse_args()
    np.random.seed(123)
    x=sample_group_parity_vec(4,12,3)
    print('Product-parity validity:',(parity_func(x.reshape(4,4,3),axis=2)==0).all(axis=1))
    task=SwitchTask(n_people=36,group_size=2,parity_target=0,dressing=0,answer_format='spaced')
    answer=format_answer(sample_valid(task,1)[0],task)
    print(render_prompt(task));print(answer);print(verify(answer,task))
    if args.images:
        from genrule.datasets.latents import sample_latents
        from genrule.datasets.images.flower import PRESETS, render_batch, decode_batch
        cfg=PRESETS['aster24'];lat=sample_latents('parity',2,cfg,seed=123,group_size=2)
        images=render_batch(lat,cfg,seed=19);rec,ok,_=decode_batch(images,cfg)
        print('Flower shapes:',lat.shape,images.shape,'readable:',ok,'exact round trip:',np.array_equal(lat,rec))

if __name__=='__main__':main()
