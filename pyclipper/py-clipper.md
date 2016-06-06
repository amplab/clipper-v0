
###Clipper Python Interface###

To use PyClipper, you must compile the Rust library backing it first:
```
$ cargo build --release
```

Once the dylib is built, you should be able to use the Python interface similar to the REST interface or calling Clipper from Rust directly.

There are a few things in `pyclipper/src/lib.rs` that are hardcoded right now that should
not be going forward.

+ Line 108: The input type is hardcoded as `InputType::Float(7)`. Change that line if you want arrays of a different length. Right now only float inputs are supported (numpy arrays), but it should be pretty easy to change that if you need to a different input.
+ Line 137: all predictions are made with uid 0, meaning predictions are not personalized right now.


Most importantly, the task model in use right now is `DummyTaskModel` (see line 32).
This is definitely not what you want. It just sums the model predictions and returns the result and is basically just a stub for debugging.
