Actually, proper warm-starting requires matching contacts across frames (same pair → reuse lambda). Without contact matching, reusing lambdas from random slots would be worse. For now just keep zeroing but the damping alone should help significantly.

Try zig build run. The 0.99 damping per frame should kill most of the jitter while barely affecting falling speed