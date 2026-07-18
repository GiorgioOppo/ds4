# Project path security

## Boundary

The imported root is the only authorized space. Every tool argument is
treated as a relative path: `..` components, absolute paths and destinations
whose standardized/resolved path escapes the root must be rejected.

## Symbolic links

Symlinks are not indexed and are never valid paths for tools, even when they
point to an internal destination. Before reading, writing, editing or
deleting, every existing component under the root is checked separately. The
check does not just resolve the final URL: if the leaf does not exist,
Foundation may leave a symlink in a parent directory unresolved.
Genuinely missing directories remain valid for the creation of new files.
Writes revalidate the path after creating the intermediate directories and
right before the I/O, so as to shrink the TOCTOU window.

## Indexing

Generated or very heavy directories are excluded, the number of files and
the indexable size are capped, and known textual extensions are accepted.
Non-indexed files can be read in ranges only through the raw paths, which
apply the same boundary checks.

## Edits

An edit re-reads the file from disk before applying the substitution,
avoiding silently overwriting changes made by the editor or by git.
Deletion is limited to files; directories and the root are not valid targets.

Every new `ProjectCache` API must preserve these invariants and have tests
for traversal, symlinks, nonexistent paths and concurrent changes.
