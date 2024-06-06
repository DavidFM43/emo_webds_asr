import torch
import datasets
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import pipeline
from transformers.modeling_utils import is_flash_attn_2_available

import os
import logging
import argparse
from utils import setup_logging

logger = logging.getLogger("transcription")

model_name = "distil-whisper/distil-large-v3"
chunk_length_s = 30
batch_size = 24
n_shards = {"train": 5070, "test": 28}

def load_model(args):
    model = pipeline(
        "automatic-speech-recognition",
        model_name,
        device=f"cuda:{args.device_id}",
        torch_dtype=torch.float16,
        model_kwargs={
            "attn_implementation": "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        },
    )
    return model


def transcribe(audio, model):
    outputs = model(
        audio,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        generate_kwargs={"task": "transcribe"},
    )
    return outputs


def get_file_size(file_path):
    if os.path.exists(file_path):
        file_size_bytes = os.path.getsize(file_path)
        # Convert bytes to megabytes (1 MB = 1,048,576 bytes)
        file_size_mb = file_size_bytes / (1024 * 1024)
    else:
        file_size_mb = 0

    return file_size_mb


def write_parquet(file_path, data_gen, schema, chunk_size=1000):
    rows = []
    writer = None

    try:
        for idx, row in enumerate(data_gen):
            rows.append(row)
            if len(rows) >= chunk_size:
                logger.info(f"Transcribing record number: {idx + 1}")
                logger.info(f"Current file size: {get_file_size(file_path):.2f} MB")
                table = pa.Table.from_pandas(pd.DataFrame(rows, columns=["__key__", "transcription"]))
                if writer is None:
                    writer = pq.ParquetWriter(file_path, schema)
                writer.write_table(table)
                rows = []  

        # Write any remaining rows
        if rows:
            table = pa.Table.from_pandas(pd.DataFrame(rows, columns=["__key__", "transcription"]))
            if writer is None:
                writer = pq.ParquetWriter(file_path, schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def main(args):
    setup_logging(output=args.output_dir, level=logging.INFO)

    logger.info(f"Loading {args.split} split in streaming mode.")
    base_url = f"https://huggingface.co/datasets/krishnakalyan3/emo_webds/resolve/main/dataset/{args.split}"
    base_url = base_url + "{i}.tar"
    logger.info(f"Total number of shards {n_shards[args.split]}")

    # Shards were loaded in lexicographic order
    indices = sorted([str(i) for i in range(n_shards[args.split])])
    logger.info(f"Starting from shard {args.start_shard}")
    indices = indices[args.start_shard:]
    urls = [base_url.format(i=i) for i in indices]

    ds = datasets.load_dataset("webdataset", data_files={args.split: urls}, split=args.split, streaming=True)
    # format for transcription pipeline
    ds = ({**x["flac"], "__key__": x["__key__"]} for x in ds)

    logger.info(f"Loading model in device {args.device_id}.")
    # load whisper model
    model = load_model(args)
    # transcription iterator
    ds = transcribe(ds, model)
    # format to write parquet
    ds = ([out["__key__"][0], out["text"]] for out in ds)
    schema = pa.schema([("__key__", pa.string()), ("transcription", pa.string())])

    write_parquet(os.path.join(args.output_dir, f"transcriptions.parquet"), ds, schema)


if __name__ == "__main__":
    description = "Transcription of emo_webds"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory to write results and logs",
        required=True
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Dataset split",
        required=True
    )
    parser.add_argument(
        "--device-id",
        type=int,
        help="Cuda device id.",
        default=0
    )
    parser.add_argument(
        "--start-shard",
        type=int,
        help="Web dataset Shard index to start data loading.",
        default=0
    )
    args = parser.parse_args()

    main(args)
