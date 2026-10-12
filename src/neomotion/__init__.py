"""neomotion: self-contained Stage-1 NEST (prompt + starting pose -> 20-frame body-20 motion).

Vendored from dmtbaby.stage1_v7 plus its minimal dependencies (topology, teacher loader,
pose-file reader, CLIP text encoder) so this folder runs without the legacy codebase.
Checkpoint format strings remain "dmtbaby.stage1.v7" for compatibility with existing runs.
"""
