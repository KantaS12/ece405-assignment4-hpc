# mixed_precision_accumulation

The fp32 accumulator with fp32 addends lands at ~10.0 as expected, while the
all-fp16 loop drifts to ~9.95 because once the running sum exceeds ~1, the
fp16 ulp is larger than 0.01 and each add is rounded back down (swamping).
The two mixed cases (fp32 accumulator with fp16 addends, implicit or via
`.type(float32)`) both recover near-fp32 accuracy: keeping the accumulator
in fp32 prevents the loss of the small increment, even though the addend
itself is already quantized to the nearest fp16 representable.
