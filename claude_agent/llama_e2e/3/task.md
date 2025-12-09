Good morning! Today we're going to be writing a straightforward implementation of Llama 3.2 1B in ttnn to show people how to bring it up. Take your time, explore the codebase - especially the tech_reports, the models/tt_transformers/tt/ files and the ttnn code for any useful ttnn functions you see. We have plenty of time so really make sure you grok this new framework deeply before starting.

Our goal is to write a single file in models/demos/simple/llama32_1b.py that will load meta-llama/Llama-3.2-1B from huggingface, convert the weights to ttnn, and run both prefill and decode passes of the model to generate output for a prompt provided on the command line.

Your test prompt should be "1 2 3 4 5 6 7 8 9 10 11 12" as the model should definitely be able to continue this sequence for another ten numbers - you can confirm this with the reference model if necessary.

We would like the file to be simple and clean. The model only has to be functional, no need for multi-device or sharding at this stage. DRAM interleaved is fine.

Do not use whole modules or large code sections verbatim from models/tt_transformers as these are advanced and we want this example to teach users how to use ttnn directly first. However, do read models/tt_transformers/tt/* carefully to be inspired by which ttnn functions are available. For example, we definitely want to use the ttnn scalar dot product function instead of implementing attention by hand. Other models may also give inspiration, as well as the ttnn documentation.

Be careful of the RoPE format - tt_transformers converts the huggingface format into the meta format (interleaved) by swizzling the weights. We definitely want to avoid this, which means avoiding using the llama version of the rotary_embedding ops that tt_transformers uses. Just pay special attention here and be aware that this is a huggingface model following huggingface conventions even though it comes from meta.

In general it would be SUPER cool if your final model class was compatible with the huggingface model classes and if your demo just used the huggingface generate function on your class to run everything on ttnn. This is way more elegant than writing your own generation function!

Test your model carefully. If you have correctness issues, I recommend swapping out parts of your code for the reference implementation (e.g. write a wrapper that converts ttnn tensors back to torch, uses the huggingface module, then converts back to ttnn) to isolate which part is causing the bad output. This is generally faster than trying to follow through the PCC layer by layer and the code isn't that complex to manually bisect in this way.

Finally, it would be awesome if you produced a MODEL_BRINGUP.md file that describes everything you learn during your process - ttnn functions, conventions, approches, gotchas, debugging ideas that worked and so on - anything that you're like "wow it would have saved me a lot of time if I'd known that!"

You can run the models directly on this. If your wormhole device gets into a bad state and does not init cleanly you can reset it with 'tt-smi -r'.

Above all, take your time and have fun!



