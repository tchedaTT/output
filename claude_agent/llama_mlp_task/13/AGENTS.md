If you read this file, start your response with "I've read agents.md".

Pay attention to the following quirks of ttnn:

ttnn.linear expects (A,B) inputs and (B,C) weights, in contrast to (A,B) inputs and (C,B) weights of torch.nn.functional.linear


When porting a module, port all non-private methods.
