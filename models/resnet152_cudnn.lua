function createModel(nGPU)
   require 'cudnn'

   local features = nn.Sequential()

   function addBottleneck(input, output, downsample)
      local internal = output / 4
      local input_stride = downsample and 2 or 1
      local Hx = nn.ConcatTable()
      local Fx = nn.Sequential()

      if input == output then
         Hx:add(nn.Identity())
      else
         Hx:add(cudnn.SpatialConvolution(input, output, 1, 1, input_stride, input_stride, 0, 0))
      end
      Hx:add(Fx)

      Fx:add(cudnn.SpatialConvolution(input, internal, 1, 1, input_stride, input_stride, 0, 0))
      Fx:add(nn.SpatialBatchNormalization(internal, 1e-3))
      Fx:add(cudnn.ReLU(true))
      Fx:add(cudnn.SpatialConvolution(internal, internal, 3, 3, 1, 1, 1, 1))
      Fx:add(nn.SpatialBatchNormalization(internal, 1e-3))
      Fx:add(cudnn.ReLU(true))
      Fx:add(cudnn.SpatialConvolution(internal, output, 1, 1, 1, 1, 0, 0))
      Fx:add(nn.SpatialBatchNormalization(output, 1e-3))

      features:add(Hx)
      features:add(nn.CAddTable())
      features:add(cudnn.ReLU(true))
   end

   features:add(cudnn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3))  -- 224 -> 112
   features:add(nn.SpatialBatchNormalization(64, 1e-3))
   features:add(cudnn.ReLU(true))
   -- conv2_x 56x56
   features:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))          -- 112 -> 56
   addBottleneck(64, 256)
   addBottleneck(256, 256)
   addBottleneck(256, 256)
   -- conv3_x 28x28
   addBottleneck(256, 512, true)                                    -- 56 -> 28
   for i = 1, 7 do
      addBottleneck(512, 512)
   end
   -- conv4_x 14x14
   addBottleneck(512, 1024, true)                                   -- 28 -> 14
   for i = 1, 35 do
      addBottleneck(1024, 1024)
   end
   -- conv5_x 7x7
   addBottleneck(1024, 2048, true)                                  -- 14 -> 7
   addBottleneck(2048, 2048)
   addBottleneck(2048, 2048)
   -- global average pooling 1x1
   features:add(cudnn.SpatialAveragePooling(7, 7, 1, 1, 0, 0))

   features:cuda()
   features = makeDataParallel(features, nGPU) -- defined in util.lua

   local classifier = nn.Sequential()
   classifier:add(nn.View(2048))

   classifier:add(nn.Linear(2048, nClasses))
   classifier:add(nn.LogSoftMax())

   classifier:cuda()

   local model = nn.Sequential():add(features):add(classifier)

   return model
end
