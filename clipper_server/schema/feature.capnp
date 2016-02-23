@0x82b0defffb5f3766;

interface Feature {
# This should be generic

  computeFeature @0 (inp: List(Float64)) -> (result: Float64);

  # computeFeatureBatch @1 (inputs: List(List(Float))) -> (result: List(Float64)); 

  # hash @2(input: Data) -> (hashId: UInt32); 

}


