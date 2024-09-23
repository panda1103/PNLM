import os
import argparse
import logging
import re
from typing import Optional, Union
from Bio import SeqIO
import numpy as np
import esm
import torch
import transformers 
import gc

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_data(input_file_name: str, bidirectional: bool = False) -> list[str]:
    """
    Prepare data from the input fasta file.
    """
    # Determine if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    m = re.search(r".fasta", input_file_name)
    
    prefix = m.group(1)
    seqs = SeqIO.parse(open(input_file_name, "r"), "fasta")
    
    # Load the pretrained model and move it to the device
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)  # Move model to GPU if available
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Disables dropout for deterministic results
    parsed_seqs = []
    i = 0
    
    # Process sequences in batches to manage memory
    for s in seqs:
        # Check if the sequence length is greater than 400
        if len(s.seq) > 400:
            print(f"Skipping sequence {s.id} with length {len(s.seq)}")
            continue  # Skip this sequence if it is longer than 400

        parsed_seqs.append(f"<|pe{i+1}|>1{str(s.seq)}2")
        sequence = [(f"{prefix}", str(s.seq))]
        batch_labels, batch_strs, batch_tokens = batch_converter(sequence)
        
        # Move tokens to the device
        batch_tokens = batch_tokens.to(device)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        
        token_representations = results["representations"][33]
        seq_embeddings = token_representations.detach().cpu().numpy().reshape(-1, 1280)
        
        # Save embeddings to disk
        np.save(f"./tuningdata/pe{i+1}.npy", seq_embeddings)
        i += 1
        
        # Clean up GPU memory
        del results, batch_tokens, batch_labels, batch_strs, token_representations, seq_embeddings
        torch.cuda.empty_cache()  # Clear cache to free up GPU memory
        gc.collect()  # Force garbage collection
        
        if bidirectional:
            parsed_seqs.append(f"2{str(s.seq)[::-1]}1")

    return parsed_seqs



def main(args: argparse.Namespace):
    np.random.seed(args.seed)
    # 数据集分割比例
    if not 0 <= args.train_split_ratio <= 1:
        raise ValueError("Train-test split ratio must be between 0 and 1.")

    # 分割数据集
    train_data = []
    test_data = []
    for input_file in args.input_files:
        data = prepare_data(input_file, args.bidirectional)
        logging.info(f"Loaded {len(data)} sequences from {input_file}")
        np.random.shuffle(data)
        split_idx = int(len(data) * args.train_split_ratio)
        train_data.extend(data[:split_idx])
        test_data.extend(data[split_idx:])
    #np.random.shuffle(train_data)
    #np.random.shuffle(test_data)

    #双向数据集，打印log
    if args.bidirectional:
        logging.info("Data is bidirectional. Each sequence will be stored in both directions.")

    logging.info(f"Train data: {len(train_data)} sequences")
    logging.info(f"Test data: {len(test_data)} sequences")

    logging.info(f"Saving training data to {args.output_file_train}")
    with open(args.output_file_train, "w") as f:
        for line in train_data:
            f.write(line + "\n")

    logging.info(f"Saving test data to {args.output_file_test}")
    with open(args.output_file_test, "w") as f:
        for line in test_data:
            f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_files", type=str, nargs="+", required=True, help="Input fasta files."
    )
    parser.add_argument(
        "--output_file_train", type=str, default="train_data.txt", help="Output file for the train data split. Default: train_data.txt"
    )
    parser.add_argument(
        "--output_file_test", type=str, default="test_data.txt", help="Output file for test data split. Default: test_data.txt"
    )
    parser.add_argument(
        "--bidirectional",
        "-b",
        action="store_true",
        help="Whether to store also the reverse of the sequences. Default: False.",
    )
    parser.add_argument(
        "--train_split_ratio",
        "-s",
        type=float,
        default=0.8,
        help="Train-test split ratio. Default: 0.8",
    )
    parser.add_argument(
        "--seed", type=int, default=69, help="Random seed",
    )
    args = parser.parse_args()
    main(args)
