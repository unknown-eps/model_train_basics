"""Write completed generations as independently readable Parquet shards."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


class ShardWriter:
    """Single-owner writer; offload add/finish to a thread from async code."""

    def __init__(self, output_dir: str | Path, shard_size: int):
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.shard_size = shard_size
        self.schema = pa.schema([
            ("id", pa.int32()),
            ("tokens", pa.list_(pa.int32())),
        ])
        self.buffer = []
        self.shards_written = 0

    def add(self, record: dict) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.shard_size:
            self._flush()

    def finish(self) -> None:
        if self.buffer:
            self._flush()

    def _flush(self) -> None:
        path = self.output_dir / f"shard-{self.shards_written:05d}.parquet"
        temporary_path = path.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        pq.write_table(table, temporary_path, compression="zstd")
        temporary_path.rename(path)
        print(f"Saved {path}: {len(self.buffer)} records", flush=True)
        self.buffer = []
        self.shards_written += 1
