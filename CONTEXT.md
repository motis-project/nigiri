# nigiri

Timetable-based public transit routing library.

## Language

### Loader directory abstraction

**Directory**:
The unit the GTFS loader reads input from: a set of named files it can list and
open.

_Avoid_: data source, folder

**Filesystem directory**:
A directory backed by a real folder on disk. Files are opened directly from
the filesystem.

_Avoid_: fs_dir

**Archive-backed directory**:
A directory whose files live inside a single archive file (today: zip), rather
than a folder. Currently supports deflate and zstd.

_Avoid_: zip folder, compressed directory

**In-memory directory**:
A directory assembled purely in memory from path-to-content entries, without a
backing archive or folder.

_Avoid_: mem directory

**File content**:
The bytes of a file inside a directory, accessed by name. A file's path inside
the directory identifies it; lookup is case-sensitive.

_Avoid_: file contents, payload

**Directory type**:
Which of the three kinds of directory an instance is: filesystem, archive-
backed, or in-memory.

_Avoid_: dir kind

**Directory hash**:
A digest of the directory as a whole, used to detect whether the input
changed. For an archive-backed directory it hashes the raw archive bytes, not
the decompressed contents.
