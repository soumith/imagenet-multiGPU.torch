function createModel(nGPU)
   require 'cudnn'
   require 'nngraph'
   paths.dofile('custom_modules/CleanGModule.lua')
   paths.dofile('custom_modules/DPTSequential.lua')

   local features = nn.DPTSequential()

   function addBottleneck(input, output, size, downsample)
      local internal = output / 4
      local input_stride = downsample and 2 or 1
      local out_size = math.floor(size / input_stride)

      local module_in = nn.Identity()()
      local conv1 = cudnn.SpatialConvolution(input, internal, 1, 1, input_stride, input_stride, 0, 0)(module_in)
      local bn1   = nn.SpatialBatchNormalization(internal, 1e-3)(conv1)
      local relu1 = cudnn.ReLU(true)(bn1)
      local conv2 = cudnn.SpatialConvolution(internal, internal, 3, 3, 1, 1, 1, 1)(relu1)
      local bn2   = nn.SpatialBatchNormalization(internal, 1e-3)(conv2)
      local relu2 = cudnn.ReLU(true)(bn2)
      local conv3 = cudnn.SpatialConvolution(internal, output, 1, 1, 1, 1, 0, 0)(relu2)

      local sum
      if input == output and not downsample then
         sum = nn.CAddTable()({module_in, conv3})
      else
         local projection = cudnn.SpatialConvolution(input, output, 1, 1, input_stride, input_stride, 0, 0)(module_in)
         sum = nn.CAddTable()({projection, conv3})
      end

      local module_out = nn.SpatialBatchNormalization(output, 1e-3)(sum)
      -- removed for now http://gitxiv.com/comments/7rffyqcPLirEEsmpX
      -- features:add(cudnn.ReLU(true))
      features:add(nn.CleanGModule({module_in}, {module_out}))
   end

   features:add(cudnn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3))  -- 224 -> 112
   features:add(nn.SpatialBatchNormalization(64, 1e-3))
   features:add(cudnn.ReLU(true))
   -- conv2_x 56x56
   features:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))          -- 112 -> 56
   addBottleneck(64, 256, 56)
   addBottleneck(256, 256, 56)
   addBottleneck(256, 256, 56)
   -- conv3_x 28x28
   addBottleneck(256, 512, 56, true)                                -- 56 -> 28
   for i = 1, 3 do
      addBottleneck(512, 512, 28)
   end
   -- conv4_x 14x14
   addBottleneck(512, 1024, 28, true)                               -- 28 -> 14
   for i = 1, 5 do
      addBottleneck(1024, 1024, 14)
   end
   -- conv5_x 7x7
   addBottleneck(1024, 2048, 14, true)                              -- 14 -> 7
   addBottleneck(2048, 2048, 7)
   addBottleneck(2048, 2048, 7)
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
