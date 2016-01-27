@0x82b0defffb5f3766;

interface Feature {
# This should be generic

  computeFeature @0 (input: Data) -> (result: Float64);

  computeFeatureBatch @1 (inputs: List(Data)) -> (result: List(Float64));

  hash @2(input: Data) -> (hashId: UInt32);

}


