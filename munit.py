import io
import os
from PIL import Image
import streamlit as st
import argparse
from tempfile import NamedTemporaryFile

from imaginaire.config import Config
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.dataset import get_test_dataloader
from imaginaire.utils.distributed import init_dist
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.utils.io import get_checkpoint as get_checkpoint
from imaginaire.utils.logging import init_logging
from imaginaire.utils.trainer import \
    (get_model_optimizer_and_scheduler, get_trainer, set_random_seed)
import imaginaire.config

STREAMLIT_SCRIPT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='configs/ampO1.yaml',
                        help='Path to the training config file.')
    parser.add_argument('--checkpoint', default='model/skenario_pertama/epoch_00217_iteration_000023000_checkpoint.pt',
                        help='Checkpoint path.')
    parser.add_argument('--output_dir', default='output',
                        help='Location to save the image outputs')
    parser.add_argument('--logdir',
                        help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        
        with NamedTemporaryFile(dir='image', suffix='.jpg') as f:
            f.write(uploaded_file.getbuffer())

            return f'{STREAMLIT_SCRIPT_FILE_PATH}\image'
              
    else:
        return None
    
def inference(image):
    args = parse_args()
    set_affinity(args.local_rank)
    set_random_seed(args.seed, by_rank=True)
    cfg = Config(args.config)
    imaginaire.config.DEBUG = args.debug
    
    cfg.test_data.test.roots = image

    if not hasattr(cfg, 'inference_args'):
        cfg.inference_args = None

    # If args.single_gpu is set to True,
    # we will disable distributed data parallel.
    if not args.single_gpu:
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank)

    # Override the number of data loading workers if necessary
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    # Create log directory for storing training results.
    cfg.date_uid, cfg.logdir = init_logging(args.config, args.logdir)

    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    # Initialize data loaders and models.
    test_data_loader = get_test_dataloader(cfg)
    net_G, net_D, opt_G, opt_D, sch_G, sch_D = \
        get_model_optimizer_and_scheduler(cfg, seed=args.seed)
    trainer = get_trainer(cfg, net_G, net_D,
                          opt_G, opt_D,
                          sch_G, sch_D,
                          None, test_data_loader)

    # Load checkpoint.
    trainer.load_checkpoint(cfg, args.checkpoint)

    # Do inference.
    trainer.current_epoch = -1
    trainer.current_iteration = -1
    trainer.test(test_data_loader, args.output_dir, cfg.inference_args, cfg.inference_args.random_style)


def main():
    st.title('Implementasi Model MUNIT')
    image = load_image()
    
    result = st.button('Run on image')
    if result and image is not None:
        st.write('Calculating results...')
        inference(image)
    

if __name__ == "__main__":
    main()