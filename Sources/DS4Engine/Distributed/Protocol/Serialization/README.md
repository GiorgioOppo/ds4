**English** | [Italiano](README.it.md)

# Protocol/Serialization

`Data+LittleEndian.swift` contains the internal primitives for appending and
reading little-endian integers with an explicit cursor.

## Flow and dependencies

This is the lowest layer of the protocol and depends only on
Foundation/DS4Core. All message codecs use it to get the same layout on every
Mac.

## Extension

Add only generic wire primitives. Every read must advance the cursor
predictably and be preceded by a check of the available bytes in the calling
decoder; do not put logic for a specific message here.
