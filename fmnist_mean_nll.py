import argparse
import os
import torch


def load_files(path):
    file_list = os.listdir(path)
    all_nlls = []
    for i, file in enumerate(file_list):
        all_nlls.append(torch.load(os.path.join(path, file), map_location='cpu'))
    return torch.cat(all_nlls).reshape(-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nll-folder', help='Path to the saved NLL values folder', required=True)
    args = parser.parse_args()

    # Load NLL values and compute mean
    all_nlls = load_files(args.nll_folder)
    print(all_nlls.mean())

if __name__ == "__main__":
    main()
