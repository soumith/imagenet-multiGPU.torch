
function createModel(nGPU)
   require 'cudnn'
   require 'ccn2'
   -- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
   -- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
   local features = nn.Sequential()
   features:add(nn.Transpose({1,4},{1,3},{1,2}))
   features:add(ccn2.SpatialConvolution(3,64,11,4,0,1,4))       -- 224 -> 55
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialMaxPooling(3,2))                   -- 55 ->  27
   features:add(ccn2.SpatialConvolution(64,192,5,1,2,1,3))       --  27 -> 27
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialMaxPooling(3,2))                   --  27 ->  13
   features:add(ccn2.SpatialConvolution(192,384,3,1,1,1,3))      --  13 ->  13
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialConvolution(384,256,3,1,1,1,3))      --  13 ->  13
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialConvolution(256,256,3,1,1,1,3))      --  13 ->  13
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialMaxPooling(3,2))                   -- 13 -> 6
   features:add(nn.Transpose({4,1},{4,2},{4,3}))

   features:cuda()
   features = makeDataParallel(features, nGPU) -- defined in util.lua

   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))

   local branch1 = nn.Concat(2)

   for i=1,nGPU do
      local s = nn.Sequential()
      s:add(nn.Dropout(0.5))
      s:add(nn.Linear(256*6*6, 4096/nGPU))
      s:add(nn.ReLU())
      branch1:add(s)
   end
   classifier:add(branch1)

   local branch2 = nn.Concat(2)
   for i=1,nGPU do
      local s = nn.Sequential()
      s:add(nn.Dropout(0.5))
      s:add(nn.Linear(4096, 4096/nGPU))
      s:add(nn.ReLU())
      branch2:add(s)
   end
   classifier:add(branch2)
   classifier:add(nn.Linear(4096, nClasses))
   classifier:add(nn.LogSoftMax())
   classifier:cuda()

   local model = nn.Sequential():add(features):add(classifier)
   model.imageSize = 256
   model.imageCrop = 224

   return model
end
