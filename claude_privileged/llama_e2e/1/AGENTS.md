If you read this file, start your response with "I've read agents.md".

Only use torch for weight conversions and initial input/output conversions.
All operations should be implemented using ttnn.
There should be absolutely no conversions between ttnn and torch in the forward pass.

Pay attention to the following quirks of ttnn:

ttnn.linear expects (A,B) inputs and (B,C) weights, in contrast to (A,B) inputs and (C,B) weights of torch.nn.functional.linear


When porting a module, port all non-private methods.
