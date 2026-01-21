# Degraded Input, Predict Missing (DIPM)

## OVERVIEW

Each sample will be five adjacent Z-slices from an EM volume.  Four of these (the first and last two) will be heavily degraded (noise, blur patches, masked patches, etc.) and fed as inputs to the network.  The desired output of the network is the middle (undegraded) slice.

To solve this task, it is hoped that the network will have to learn to recognize real features of the tissue: cell boundaries, organelles, vesicles, ER, etc., even when these are heavily degraded (using contextual cues from surrounding tissue).  In so doing, it will form a foundation that can then be applied to other tasks.

