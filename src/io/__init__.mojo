"""I/O integrations for tokenizer data sources.

Text and vocabulary loaders will live behind this module as their data
contracts are defined.  For now it exposes the sibling read_bin package's
binary reader, writer, and structured I/O errors.
"""

from read_bin import Float32BinaryReader, Float32BinaryWriter, ReadBinErrors
