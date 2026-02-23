## What changes in batch_size do:
- Smaller batches (16-32): More weight updates, noisier gradients, better generalization, slower
- Larger batches (128-512): Fewer updates, smoother gradients, faster per epoch, may need higher LR