# minizip-ng for the zip directory

The loader's archive-backed directory needs to read zip archives in more
compression formats than deflate. We swapped miniz for minizip-ng (pinned
4.2.2) as the implementation behind the zip directory. miniz only reads
deflate; minizip-ng ships zstd support already (enabled in this change) and
adding further formats later (e.g. bzip2, xz, AES encryption) is a dependency
entry plus one build flag, never a code change. The trade-off is a bigger
dependency footprint (minizip-ng plus zstd) versus format coverage; both are
MIT-compatible. Deflate itself keeps coming from zlib-ng already in the tree,
so only zstd is enabled today.

File-backed archives are opened through the library's own file stream (no
mmap), while in-memory archives are handed over zero-copy via
`mz_zip_reader_open_buffer(..., copy=0)` over the caller's buffer.

Unchanged by this decision: the public `dir`/`file` API, the pimpl boundary,
case-sensitive lookup, the `.zip` extension dispatch in `make_dir`, and the
`hash()` semantics of hashing the raw archive bytes. The existing deflate
fixture tests (cross-checking zip, file and memory directories) act as the
parity gate.
