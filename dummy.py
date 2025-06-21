import os
import torch
from train import main
import sys

def test_with_dummy():
    # Create dummy dataset
    dummy_dir = "dummy_data"
    os.makedirs(f"{dummy_dir}/train/input", exist_ok=True)
    os.makedirs(f"{dummy_dir}/train/target", exist_ok=True)
    
    # Generate fake images
    for i in range(10):
        torch.save(torch.rand(3, 256, 256), f"{dummy_dir}/train/input/{i}.pt")
        torch.save(torch.rand(3, 256, 256), f"{dummy_dir}/train/target/{i}.pt")

    # Test both models
    for model_name in ['edge', 'lied']:
        print(f"\n=== Testing {model_name} model ===")
        
        # Save original sys.argv
        original_argv = sys.argv
        
        # Set up fake command-line arguments
        sys.argv = [
            'dummy.py',  # Program name
            '--model', model_name,
            '--config_yaml', 'config.yml',
            *[f'TRAINING.TRAIN_DIR={dummy_dir}/train'],
            *['OPTIM.BATCH_SIZE=2'],
            *['OPTIM.NUM_EPOCHS=1'],
            *['TRAINING.VAL_AFTER_EVERY=1']
        ]
        
        try:
            main()  # Now it will use our fake command-line args
        finally:
            # Restore original sys.argv
            sys.argv = original_argv

if __name__ == '__main__':
    test_with_dummy()